import torch.nn as nn


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()

    def predict(self, image, model):
        # TODO implement this properly using the sample methods from Evaluator
        return image
