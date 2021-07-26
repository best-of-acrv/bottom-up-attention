import acrv_datasets
import numpy as np
from PIL import Image
import os
import torch

from .datasets import helpers as dh
from .datasets.captioning import CaptionDataset
from .datasets.vqa import VqaDataset


class BottomUpAttention(object):
    DATASETS = ['captions', 'vqa']
    TASKS = ['captioning', 'vqa']

    def __init__(self,
                 *,
                 cache_dir=os.path.expanduser('~/.bottom-up-attention-cache'),
                 gpu_id=0,
                 load_pretrained=None,
                 load_snapshot=None,
                 model_name='bottom-up-attention',
                 model_seed=0,
                 num_hidden_dims=1024,
                 task='captioning'):
        # Apply sanitised arguments
        self.cache_dir = cache_dir
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.num_hidden_dims = num_hidden_dims
        self.task = _sanitise_arg(task, 'task', BottomUpAttention.TASKS)

        # Ensure cache exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Try setting up GPU integration
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.manual_seed(self.gpu_id)
        torch.cuda.manual_seed(self.gpu_id)
        # torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise RuntimeWarning('PyTorch could not find CUDA, using CPU ...')

        # Load the model based on the specified parameters
        self.model = None
        if self.load_snapshot:
            print("\nLOADING MODEL FROM SNAPSHOT:")
            self.model = _from_snapshot(self.load_snapshot)
        else:
            print("\nLOADING MODEL FROM PRE-TRAINED WEIGHTS:")
            self.model = _get_model(self.num_hidden_dims)
        self.model.cuda()

    def evaluate(self, *, dataset_dir=None, output_directory='./eval_output'):
        # Load in the dataset
        dataset = _load_dataset(self.task, dataset_dir, 'eval')

        # Perform the requested evaluation
        e = Evaluator(output_directory=output_directory)
        fn = e.sample_vqa if self.task == 'vqa' else e.sample_captioning
        fn(self.model, dataset)

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
        out = Predictor().predict(
            img,
            self.model,
        )

        # Save the file if requested, & return the output
        if output_file:
            Image.fromarray(out).save(output_file)
        return out

    def train(self,
              *,
              dataset_dir=None,
              display_interval=100,
              eval_interval=1,
              learning_rate=2e-3,
              max_epochs=50,
              num_workers=1,
              output_directory=os.path.expanduser(
                  '~/bottom-up-attention-output'),
              snapshot_interval=5):
        # Load in the dataset
        dataset = _load_dataset(self.task, dataset_dir, self.cache_dir,
                                'train')

        # Start a model trainer
        print("\nPERFORMING TRAINING:")
        Trainer(output_directory).train(self.model)


def _load_dataset(task, dataset_dir, mode, cache_dir, quiet=False):
    # Print some verbose information
    if not quiet:
        print("\nGETTING DATASET:")
    if dataset_dir is None:
        dataset_dir = acrv_datasets.get_datasets_directory()
    if not quiet:
        print("Using 'dataset_dir': %s" % dataset_dir)

    # Defer to acrv_datasets for making sure required datasets are downloaded
    datasets = [
        'coco/vqa_questions_train', 'coco/vqa_questions_test',
        'coco/vqa_questions_val', 'glove', 'caption_features/trainval2014_36'
    ]
    dirs = acrv_datasets.get_datasets(datasets, datasets_directory=dataset_dir)
    if not quiet:
        print("\n")

    # Ensure all required derived data exists
    # TODO glue this together
    fn_dictionary = os.path.join(cache_dir, 'dictionary.pkl')
    fn_embeddings = os.path.join(cache_dir, 'glove6b_init.npy')
    dh.make_dictionary(dirs[:3], fn_dictionary)
    dh.make_glove_embeddings(fn_dictionary, dirs[3], fn_embeddings)
    dh.compute_softscore()
    dh.convert_detection_features()
    dh.create_caption_input_data()

    # Return a PyTorch dataset with the appropriate wrappings
    # TODO


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
