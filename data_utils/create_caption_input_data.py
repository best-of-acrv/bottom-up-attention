import os
import json
from tqdm import tqdm
from collections import Counter
from random import choice, sample
import pickle


def create_input_files(dataset, karpathy_json_path, captions_per_image, min_word_freq, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset. Since bottom up features only available for coco, we use only coco
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    train36_folder_path = os.path.join('data', 'trainval_36')
    with open(os.path.join(train36_folder_path, 'train36_imgid2idx.pkl'), 'rb') as j:
        train_data = pickle.load(j)

    with open(os.path.join(train36_folder_path, 'val36_imgid2idx.pkl'), 'rb') as j:
        val_data = pickle.load(j)

    # Read image paths and captions for each image
    train_image_ids = []
    val_image_ids = []
    test_image_ids = []
    train_image_captions = []
    val_image_captions = []
    test_image_captions = []
    train_image_det = []
    val_image_det = []
    test_image_det = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        filename = img['filename']
        image_id = img['filename'].split('_')[2]
        image_id = int(image_id.lstrip("0").split('.')[0])

        if img['split'] in {'train', 'restval'}:
            if img['filepath'] == 'train2014':
                if image_id in train_data:
                    train_image_det.append(("t", train_data[image_id]))
            else:
                if image_id in val_data:
                    train_image_det.append(("v", val_data[image_id]))
            train_image_captions.append(captions)
            train_image_ids.append(filename)
        elif img['split'] in {'val'}:
            if image_id in val_data:
                val_image_det.append(("v", val_data[image_id]))
            val_image_captions.append(captions)
            val_image_ids.append(filename)
        elif img['split'] in {'test'}:
            if image_id in val_data:
                test_image_det.append(("v", val_data[image_id]))
            test_image_captions.append(captions)
            test_image_ids.append(filename)

    # Sanity check
    assert len(train_image_det) == len(train_image_captions) == len(train_image_ids)
    assert len(val_image_det) == len(val_image_captions) == len(val_image_ids)
    assert len(test_image_det) == len(test_image_captions) == len(test_image_ids)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    for impaths, imcaps, imids, split in [(train_image_det, train_image_captions, train_image_ids, 'TRAIN'),
                                   (val_image_det, val_image_captions, val_image_ids, 'VAL'),
                                   (test_image_det, test_image_captions, test_image_ids, 'TEST')]:
        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)

    # Save bottom up features indexing to JSON files
    with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_image_det, j)

    with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_image_det, j)

    with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_image_det, j)

    # Save image IDs to JSON files
    with open(os.path.join(output_folder, 'TRAIN' + '_IMAGE_IDS_' + base_filename + '.json'), 'w') as j:
        json.dump(train_image_ids, j)

    with open(os.path.join(output_folder, 'VAL' + '_IMAGE_IDS_' + base_filename + '.json'), 'w') as j:
        json.dump(val_image_ids, j)

    with open(os.path.join(output_folder, 'TEST' + '_IMAGE_IDS_' + base_filename + '.json'), 'w') as j:
        json.dump(test_image_ids, j)

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path=os.path.join('data', 'mscoco', 'caption_datasets', 'dataset_coco.json'),
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=os.path.join('data', 'mscoco', 'caption_datasets'),
                       max_len=50)