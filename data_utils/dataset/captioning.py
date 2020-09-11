import os
import torch
from torch.utils.data import Dataset
import h5py
import json


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, name, dataroot='data', data_name='coco_5_cap_per_img_5_min_word_freq', transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = name.upper()
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # image directory
        if self.split == 'TEST':
            name = 'val'
        self.image_prefix = os.path.join('data', 'mscoco', name + '2014')

        # Open hdf5 file where image features are stored
        self.train_hf = h5py.File(os.path.join(dataroot, 'trainval_36', 'train36.hdf5'), 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(os.path.join(dataroot, 'trainval_36', 'val36.hdf5'), 'r')
        self.val_features = self.val_hf['image_features']

        # Captions per image
        self.cpi = 5

        # Load encoded captions
        with open(os.path.join(dataroot, 'mscoco', 'caption_datasets', self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        with open(os.path.join(dataroot, 'mscoco', 'caption_datasets', self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(dataroot, 'mscoco', 'caption_datasets', self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # Load image ids
        with open(os.path.join(dataroot, 'mscoco', 'caption_datasets', self.split + '_IMAGE_IDS_' + data_name + '.json'), 'r') as j:
            self.imageids = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.v_dim = self.train_features.shape[2]
        self.dataset_size = len(self.captions)

    def __getitem__(self, index):

        # The Nth caption corresponds to the (N // captions_per_image)th image
        objdet = self.objdet[index // self.cpi]
        imageid = self.imageids[index // self.cpi]

        # Load bottom up image features
        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])

        caption = torch.LongTensor(self.captions[index])
        caplen = torch.LongTensor([self.caplens[index]])

        # create sample with question, answer, labels and score
        sample = {'features': img, 'caption': caption, 'caption_len': caplen, 'imageid': imageid}

        if self.split != 'TRAIN':
            # For validation or testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((index // self.cpi) * self.cpi):(((index // self.cpi) * self.cpi) + self.cpi)])
            sample['all_captions'] = all_captions

        return sample

    def __len__(self):
        return self.dataset_size