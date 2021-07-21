import os
import torch


class BottomUpAttention(object):
    DATASETS = ['captions', 'vqa']
    TASKS = ['captioning', 'vqa']

    def __init__(self,
                 *,
                 gpu_id=0,
                 load_pretrained=None,
                 load_snapshot=None,
                 model_name='bottom-up-attention',
                 model_seed=0,
                 num_hidden_dims=1024,
                 task='captioning'):
        # Apply sanitised arguments
        self.gpu_id = gpu_id
        self.model_seed = model_seed
        self.num_hidden_dims = num_hidden_dims
        self.task = _sanitise_arg(task, 'task', BottomUpAttention.TASKS)

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
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        pass

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
        # Perform argument validation / set defaults
        dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                     BottomUpAttention.DATASETS)

        # Load in the dataset
        dataset = _load_dataset(self.task, dataset_dir, 'train')


def _load_dataset(task, dataset_dir, mode):
    pass


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
