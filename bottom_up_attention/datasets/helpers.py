import glob
import json
import numpy as np
import os

from .vqa import Dictionary


def compute_softscore():
    pass


def convert_detection_features():
    pass


def create_caption_input_data():
    pass


def make_dictionary(data_dirs, output_file):
    FILENAMES = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    jsons = []
    for d in data_dirs:
        jsons.extend(glob.glob(os.path.join(d, '**/*.json'), recursive=True))

    d = Dictionary()
    for f in FILENAMES:
        fn = next(x for x in jsons if x.endswith(f))
        for q in json.load(open(fn))['questions']:
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
