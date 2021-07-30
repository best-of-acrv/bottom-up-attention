import acrv_datasets
import json
import numpy as np
import os
from PIL import Image
import torch

from .evaluator import Evaluator
from .datasets import helpers as dh
from .datasets.captioning import CaptionDataset
from .datasets.vqa import Dictionary, VqaDataset
from .model import captioning_model
from .model import vqa_model
from .predictor import Predictor
from .trainer import Trainer


class BottomUpAttention(object):
    TASKS = ['captioning', 'vqa']

    def __init__(self,
                 *,
                 cache_dir=os.path.expanduser('~/.bottom-up-attention-cache'),
                 dataset_dir=None,
                 gpu_id=0,
                 load_pretrained=True,
                 load_snapshot=None,
                 load_snapshot_optimiser=True,
                 model_name='bottom-up-attention',
                 model_seed=0,
                 num_hidden_dims=1024,
                 task='captioning'):
        # Apply sanitised arguments
        self.cache_dir = cache_dir
        self.dataset_dir = dataset_dir
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.num_hidden_dims = num_hidden_dims
        self.task = _sanitise_arg(task, 'task', BottomUpAttention.TASKS)

        self.load_pretrained = load_pretrained
        self.load_snapshot = load_snapshot
        self.load_snapshot_optimiser = load_snapshot_optimiser

        # Ensure cache exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Try setting up GPU integration
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.manual_seed(self.model_seed)
        torch.cuda.manual_seed(self.model_seed)
        # torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeWarning('PyTorch could not find CUDA, using CPU ...')

        # The model depends on derived data obtained through load_dataset...
        # ideally we would re-write the models so they don't depend on data
        # they shouldn't need at initialisaiton time, but instead we lazily
        # load the dataset here as well...
        # TODO fix this!
        dataset = _load_dataset(self.task, self.dataset_dir, 'eval',
                                self.cache_dir)

        # Load the model based on the specified parameters
        print("\nLOADING MODEL FROM %s:" %
              ('SNAPSHOT' if load_snapshot else
               'PRE-TRAINED WEIGHTS' if self.load_pretrained else 'SCRATCH'))
        self.model = _get_model(self.task, dataset, self.cache_dir,
                                self.num_hidden_dims, self.load_pretrained,
                                self.load_snapshot,
                                self.load_snapshot_optimiser)
        self.model.cuda()

    def evaluate(self, *, batch_size=100, output_directory='./eval_output'):
        # Load in the dataset
        dataset = _load_dataset(self.task, self.dataset_dir, 'eval',
                                self.cache_dir)

        # Perform the requested evaluation
        Evaluator(batch_size=batch_size,
                  output_directory=output_directory).sample(
                      self.task, self.model, dataset)

    def predict(self, *, image=None, image_file=None, output_file=None):
        # Handle input arguments
        if image is None and image_file is None:
            raise ValueError("Only one of 'image' or 'image_file' can be "
                             "used in a call, not both.")
        elif image is not None and image_file is not None:
            raise ValueError("Either 'image' or 'image_file' must be provided")
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Construct the input image
        img = (np.array(Image.open(image_file).convert('RGB'))
               if image_file else image)

        # Perform the forward pass
        # TODO create Predictor class... (look in Evaluator)
        out = Predictor().predict(img, self.model)

        # Save the file if requested, & return the output
        if output_file:
            Image.fromarray(out).save(output_file)
        return out

    def train(self,
              *,
              batch_size=100,
              display_interval=100,
              eval_interval=1,
              learning_rate=2e-3,
              max_epochs=50,
              output_directory=os.path.expanduser(
                  '~/bottom-up-attention-output'),
              snapshot_interval=5):
        # Load in the dataset
        dataset = _load_dataset(self.task, self.dataset_dir, self.cache_dir,
                                'train')

        # Start a model trainer
        print("\nPERFORMING TRAINING:")
        self.model.create_optimiser(learning_rate=learning_rate)
        Trainer(output_directory).train(self.model,
                                        self.task,
                                        dataset,
                                        batch_size=batch_size,
                                        display_interval=display_interval,
                                        eval_interval=eval_interval,
                                        max_epochs=max_epochs,
                                        snapshot_interval=snapshot_interval)


def _get_model(task,
               dataset,
               cache_dir,
               number_hidden_dims,
               load_pretrained,
               snapshot_filename=None,
               use_snapshot_optimiser=True):
    m = (captioning_model.baseline if task == BottomUpAttention.TASKS[0] else
         vqa_model.baseline)(dataset,
                             number_hidden_dims,
                             pretrained=load_pretrained)
    if task == BottomUpAttention.TASKS[1]:
        m.w_emb.init_embedding(os.path.join(cache_dir, 'glove6b_init.npy'))

    model_data = None
    if snapshot_filename is not None:
        print('Loading model from:\n\t%s' % snapshot_filename)
        model_data = torch.load(snapshot_filename)
        m.load_state_dict(model_data['weights'])

    m.create_optimiser(optimiser_state=(
        None if model_data is None or not use_snapshot_optimiser else
        model_data['optimiser']))

    return m


def _load_dataset(task, dataset_dir, mode, cache_dir, quiet=False):
    # Print some verbose information
    if not quiet:
        print("\nGETTING DATASET:")
    if dataset_dir is None:
        dataset_dir = acrv_datasets.get_datasets_directory()
    if not quiet:
        print("Using 'dataset_dir': %s" % dataset_dir)

    # Defer to acrv_datasets for making sure required datasets are downloaded
    DATASETS = [
        'coco/vqa_questions_train', 'coco/vqa_questions_test',
        'coco/vqa_questions_val', 'coco/vqa_annotations_train',
        'coco/vqa_annotations_val', 'glove', 'caption_features/trainval2014_36'
    ]
    ds = {
        d: f for d, f in zip(
            DATASETS,
            acrv_datasets.get_datasets(DATASETS,
                                       datasets_directory=dataset_dir))
    }
    if not quiet:
        print("\n")

    # Ensure all required derived data exists
    fn_dictionary = os.path.join(cache_dir, 'dictionary.pkl')
    fn_embeddings = os.path.join(cache_dir, 'glove6b_init.npy')
    fn_train_hd5 = os.path.join(cache_dir, 'train36.hdf5')
    fn_val_hd5 = os.path.join(cache_dir, 'val36.hdf5')
    fn_train_indices = os.path.join(cache_dir, 'train36_imgid2idx.pkl')
    fn_val_indices = os.path.join(cache_dir, 'val36_imgid2idx.pkl')

    dh.make_dictionary([v for k, v in ds.items() if 'vqa_questions' in k],
                       fn_dictionary)
    dh.make_glove_embeddings(fn_dictionary, ds['glove'], fn_embeddings)
    dh.generate_softscores(
        [v for k, v in ds.items() if 'vqa_annotations' in k], cache_dir)
    dh.generate_detection_features(ds['caption_features/trainval2014_36'],
                                   fn_train_hd5, fn_val_hd5, fn_train_indices,
                                   fn_val_indices)
    caption_name = dh.make_caption_input_data(
        ds['caption_features/trainval2014_36'], fn_train_indices,
        fn_val_indices, cache_dir)

    # Load any required PyTorch dataset with the appropriate wrappings
    train_dataset = None
    test_dataset = None
    eval_dataset = None
    mode = mode.lower()
    if task == BottomUpAttention.TASKS[0]:
        # Captioning
        with open(os.path.join(cache_dir, 'WORDMAP_%s.json' % caption_name),
                  'r') as f:
            word_map = json.load(f)
        if mode == 'train':
            train_dataset = CaptionDataset(mode, cache_dir, caption_name)
            train_dataset.word_map = word_map
        elif mode == 'test':
            # TODO when is this ever even going to be called?!?!
            train_dataset = CaptionDataset(mode, cache_dir, caption_name)
            train_dataset.word_map = word_map
        eval_dataset = CaptionDataset(mode, cache_dir, caption_name)
        eval_dataset.word_map = word_map
    elif task == BottomUpAttention.TASKS[1]:
        # VQA
        d = Dictionary.load_from_file(fn_dictionary)
        if mode == 'train':
            train_dataset = VqaDataset('train', d)
        eval_dataset = VqaDataset('val', d)

    # Return a dataset bundle with all found datasets
    return {
        **({} if train_dataset is None else {
               'train': train_dataset
           }),
        **({} if test_dataset is None else {
               'test': test_dataset
           }),
        **({} if eval_dataset is None else {
               'val': eval_dataset
           }),
    }


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
