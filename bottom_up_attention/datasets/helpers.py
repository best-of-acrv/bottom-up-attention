import base64
import csv
import glob
import h5py
import json
import numpy as np
import os
import pickle
import re

from .vqa import Dictionary

ARTICLES = ['a', 'an', 'the']

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've"
}

DIGIT_MAP = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10'
}

PUNCTUATION = [
    ';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
    '>', '<', '@', '`', ',', '?', '!'
]

STRIP_PERIOD = re.compile(r'(?!<=\d)(\.)(?!\d)')
STRIP_COMMA = re.compile(r'(\d)(\,)(\d)')


def _generate_targets(answers_dset, ans2label, name, output_dir):
    SCORES = {0: 0, 1: 0.3, 2: 0.6, 3: 0.9}

    target = []
    for ans_entry in answers_dset:
        answers = ans_entry['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer['answer']
            answer_count[answer_] = answer_count.get(answer_, 0) + 1

        labels = []
        scores = []
        for answer in answer_count:
            if answer not in ans2label:
                continue
            labels.append(ans2label[answer])
            scores.append(SCORES.get(answer_count[answer], 1))

        target.append({
            'question_id': ans_entry['question_id'],
            'image_id': ans_entry['image_id'],
            'labels': labels,
            'scores': scores
        })

    pickle.dump(target,
                open(os.path.join(output_dir, name + '_target.pkl'), 'wb'))
    return target


def _init_feature_h5py(output_filename, ids_count):
    FEATURE_LENGTH = 2048
    NUM_FIXED_BOXES = 36

    f = h5py.File(output_filename, 'w')
    return (f,
            f.create_dataset('image_features',
                             (ids_count, NUM_FIXED_BOXES, FEATURE_LENGTH),
                             'f'),
            f.create_dataset('image_bb', (ids_count, NUM_FIXED_BOXES, 4), 'f'),
            f.create_dataset('spatial_features',
                             (ids_count, NUM_FIXED_BOXES, 6), 'f'))


def _list_by_extension(search_dirs, ext='json'):
    fs = []
    for d in search_dirs:
        fs.extend(glob.glob(os.path.join(d, '**/*.%s' % ext), recursive=True))
    return fs


def _load_from_jsons(jsons, filename):
    return json.load(open(_select_file_from_list(jsons, filename)))


def _preprocess_answer(answer):
    return _process_digit_article(_process_punctuation(answer)).replace(
        ',', '')


def _process_punctuation(inText):
    outText = inText
    for p in PUNCTUATION:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(STRIP_COMMA, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = STRIP_PERIOD.sub("", outText, re.UNICODE)
    return outText


def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = DIGIT_MAP.setdefault(word, word)
        if word not in ARTICLES:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in CONTRACTIONS:
            outText[wordId] = CONTRACTIONS[word]
    outText = ' '.join(outText)
    return outText


def _select_file_from_list(files, filename):
    return next(f for f in files if f.endswith(filename))


def generate_detection_features(trainval36_dir, output_train_hdf5,
                                output_val_hdtsv_filef5, output_train_indices,
                                output_val_indices):
    FIELD_NAMES = [
        'image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features'
    ]

    # TODO load train_imgids and val_imgids???
    train_imgids = []
    val_imgids = []
    train_indices = {}
    val_indices = {}

    (train, train_img_features, train_img_bb,
     train_spatial_img_features) = _init_feature_h5py(output_train_hdf5,
                                                      len(train_imgids))
    (val, val_img_features, val_img_bb,
     val_spatial_img_features) = _init_feature_h5py(output_val_hdf5,
                                                    len(val_imgids))

    train_counter = 0
    val_counter = 0
    print("reading tsv...")
    with open(
            _select_file_from_list(_list_by_extension(trainval36_dir, 'tsv'),
                                   'genome_36.tsv'), 'r') as f:
        r = csv.DictReader(f, delimiter='\t', fieldnames=FIELD_NAMES)
        for i in r:
            i['num_boxes'] = int(i['num_boxes'])
            image_id = int(i['image_id'])
            image_w = float(i['image_w'])
            image_h = float(i['image_h'])
            bboxes = np.frombuffer(
                base64.decodebytes(i['boxes'].encode('utf_8')),
                dtype=np.float32).reshape((i['num_boxes'], -1))

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
                (scaled_x, scaled_y, scaled_x + scaled_width,
                 scaled_y + scaled_height, scaled_width, scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.decodebytes(i['features'].encode('utf_8')),
                    dtype=np.float32).reshape((i['num_boxes'], -1))
                train_spatial_img_features[
                    train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.decodebytes(i['features'].encode('utf_8')),
                    dtype=np.float32).reshape((i['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')
    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    train.close()
    val.close()
    with open(output_train_indices, 'wb') as f:
        pickle.dump(train_indices, f)
    with open(output_val_indices, 'wb') as f:
        pickle.dump(val_indices, f)
    print("Done!")


def generate_softscores(data_dirs, output_dir):
    # Load all required data from the data directories
    jsons = _list_by_extension(data_dirs)
    train_answers = _load_from_jsons(jsons,
                                     'v2_mscoco_train2014_annotations.json')
    val_answers = _load_from_jsons(jsons, 'v2_mscoco_val2014_annotations.json')
    answers = train_answers + val_answers

    # Filter out answers that don't appear at least 9 times
    # (no idea why 9...)
    MIN_OCCURENCE = 9
    occurence = {}
    for a in answers:
        gtruth = _preprocess_answer(a['multiple_choice_answer'])
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(a['question_id'])
    for a in list(occurence):
        if len(occurence[a]) < MIN_OCCURENCE:
            occurence.pop(a)
    print('Num of answers that appear >= %d times: %d' %
          (MIN_OCCURENCE, len(occurence)))

    # Create ans2label & label2ans pickles for trainval
    NAME = 'trainval'
    ans2label = {}
    label2ans = []
    label = 0
    for a in occurence:
        label2ans.append(a)
        ans2label[a] = label
        label += 1
    pickle.dump(ans2label,
                open(os.path.join(output_dir, NAME + '_ans2label.pkl'), 'wb'))
    pickle.dump(label2ans,
                open(os.path.join(output_dir, NAME + '_label2ans.pkl'), 'wb'))

    # Generate targets for train & val
    _generate_targets(train_answers, ans2label, 'train', output_dir)


def make_caption_input_data(cache_dir, captions_dir, output_dir):
    # Read in data
    data = _load_from_jsons(_list_by_extension(captions_dir),
                            'dataset_coco.json')
    pass


def make_dictionary(data_dirs, output_file):
    FILENAMES = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    jsons = _list_by_extension(data_dirs)
    d = Dictionary()
    for f in FILENAMES:
        for q in _load_from_jsons(jsons, f)['questions']:
            d.tokenize(q['question'], True)

    d.dump_to_file(output_file)


def make_glove_embeddings(dictionary_file, glove_dir, output_file):
    # Load entries from appropriate glove file
    EMBEDDING_DIM = 300
    with open(os.path.join(glove_dir, 'glove.6B.%d.txt'), 'r') as f:
        entries = f.readlines()

    # Create dictionary of words & their embeddings, and generate weights
    d = Dictionary.load_from_file(dictionary_file)
    word2emb = {
        e.split(' ')[0]: np.array([float(i) for i in e.split(' ')[1:]
                                  ]) for e in entries
    }
    weights = np.zeros((len(d.idx2word), EMBEDDING_DIM), dtype=np.float32)
    for i, word in enumerate(d.idx2word):
        if word in word2emb:
            weights[i] = word2emb[word]

    # Write the results to file
    np.save(output_file, weights)
