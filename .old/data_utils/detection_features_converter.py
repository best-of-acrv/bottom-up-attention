"""
Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
import os
import argparse
import sys
import base64
import csv
import h5py
import pickle
import numpy as np

# get general arguments
parser = argparse.ArgumentParser(description='Detection Features Converter')
# add dataset specific arguments
parser.add_argument('--data_directory', type=str, default='../acrv-datasets/datasets', help='root directory of datasets')
args = parser.parse_args()

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

infile = os.path.join(args.data_directory, 'trainval36', 'trainval_resnet101_faster_rcnn_genome_36.tsv')

train_data_file = os.path.join(args.data_directory, 'trainval36', 'train36.hdf5')
val_data_file = os.path.join(args.data_directory, 'trainval36', 'val36.hdf5')

train_indices_file = os.path.join(args.data_directory, 'trainval36', 'train36_imgid2idx.pkl')
val_indices_file = os.path.join(args.data_directory, 'trainval36', 'val36_imgid2idx.pkl')

train_ids_file = os.path.join('data', 'train_ids.pkl')
val_ids_file = os.path.join('data', 'val_ids.pkl')

feature_length = 2048
num_fixed_boxes = 36


if __name__ == '__main__':

    # create h5py files to save train and val data
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")
    train_imgids = pickle.load(open(train_ids_file, 'rb'))
    val_imgids = pickle.load(open(val_ids_file, 'rb'))
    train_indices = {}
    val_indices = {}

    # train datasets in h5py files
    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    # train datasets in h5py files
    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    train_counter = 0
    val_counter = 0

    print("reading tsv...")
    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['num_boxes'] = int(item['num_boxes'])
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            boxes = str.encode(item['boxes'])
            bboxes = np.frombuffer(base64.decodebytes(item['boxes'].encode('utf_8')), dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.decodebytes(item['features'].encode('utf_8')),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(base64.decodebytes(item['features'].encode('utf_8')), dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    pickle.dump(train_indices, open(train_indices_file, 'wb'))
    pickle.dump(val_indices, open(val_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    print("done!")
