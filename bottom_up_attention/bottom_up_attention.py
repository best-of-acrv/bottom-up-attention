import os


class BottomUpAttention(object):
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
        print("Created instance...")

    def evaluate(self,
                 dataset_name,
                 *,
                 dataset_dir=None,
                 output_directory='./eval_output'):
        pass

    def predict(self, *, image=None, image_file=None, output_file=None):
        pass

    def train(self,
              dataset_name,
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
        pass
