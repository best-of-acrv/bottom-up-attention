import os
import json
from data_utils.dataset.captioning import CaptionDataset
from data_utils.dataset.vqa import Dictionary, VQAFeatureDataset

def get_dataset(dataset_name, mode='train'):

    if dataset_name.lower() == 'captioning':
        data_name = 'coco_5_cap_per_img_5_min_word_freq'
        if mode.lower() == 'train':
            train_dataset = CaptionDataset('train', data_name=data_name)
        elif mode.lower() == 'test':
            test_dataset = CaptionDataset('test', data_name=data_name)
        eval_dataset = CaptionDataset('val')

        # Read word map
        word_map_file = os.path.join('data', 'mscoco', 'caption_datasets', 'WORDMAP_' + data_name + '.json')
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)

        # Attach word map to datasets
        if mode.lower() == 'train':
            train_dataset.word_map = word_map
        elif mode.lower() == 'test':
            test_dataset.word_map = word_map
        eval_dataset.word_map = word_map

    elif dataset_name.lower() == 'vqa':
        dictionary_path = os.path.join('data', 'mscoco', 'dictionary.pkl')
        dictionary = Dictionary.load_from_file(dictionary_path)
        if mode.lower() == 'train':
            train_dataset = VQAFeatureDataset('train', dictionary)
        eval_dataset = VQAFeatureDataset('val', dictionary)

    # Get dataset (train and evaluation) with epoch stages
    dataset = {}

    # training and testing dataset if mode is enabled
    if mode.lower() == 'train':
        dataset['train'] = train_dataset
    elif mode.lower() == 'test' and dataset_name.lower() == 'captioning':
        dataset['test'] = test_dataset
    dataset['val'] = eval_dataset

    return dataset