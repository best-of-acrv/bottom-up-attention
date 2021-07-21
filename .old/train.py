import os
import torch
from helpers.arguments import get_argument_parser
from helpers.trainer import Trainer
from model.get_model import get_model
from data_utils.get_dataset import get_dataset

# get general arguments
parser = get_argument_parser()
# add dataset specific arguments
parser.add_argument('--name', type=str, default='bottom-up-attention', help='custom prefix for naming model')
parser.add_argument('--task', type=str, default='captioning', help='type of dataset: [captioning, vqa]')
parser.add_argument('--max_epochs', type=int, default=50, help='maximum number of training epochs')
parser.add_argument('--dataroot', type=str, default='../acrv-datasets/datasets/', help='root directory of data')
parser.add_argument('--data_directory', type=str, default='../acrv-datasets/datasets', help='root directory of datasets')
parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
parser.add_argument('--load_directory', type=str, default=None, help='load model directory')
parser.add_argument('--snapshot_num', type=int, default=None, help='snapshot number of model (if any)')
args = parser.parse_args()
args.save_directory = os.path.join(args.save_directory, args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    # Get dataset (train and validation)
    dataset = get_dataset(dataset_name=args.task, data_root=args.dataroot, mode='train')

    # Get corresponding model for task
    model = get_model(args, dataset['train'], pretrained=False)

    # if GPUs are available, move model to cuda
    if model.cuda_available:
        model = model.cuda()

    # try to load model (if any)
    if args.load_directory:
        model.load(log_directory=args.load_directory, snapshot_num=args.snapshot_num)

    # initialise model trainer and train
    trainer = Trainer(args)
    if args.task == 'captioning':
        trainer.train_captioning(args, model, dataset)
    elif args.task == 'vqa':
        trainer.train_vqa(args, model, dataset)



