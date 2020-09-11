import os
import json
import numpy as np
from data_utils.dataset.vqa import Dictionary

def create_dictionary(dataroot):
    dictionary = Dictionary()
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    # create dictionary of words and their embeddings
    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = [float(i) for i in vals[1:]]
        word2emb[word] = np.array(vals)

    # create embedding weight matrix
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    # create dictionary pickle from MSCOCO annotations
    dataroot = os.path.join('data', 'mscoco')
    d = create_dictionary(dataroot)
    d.dump_to_file(os.path.join(dataroot, 'dictionary.pkl'))

    # load dictionary pickle and create embeddings
    d = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))
    emb_dim = 300

    # create glove embeddings
    glove_file = os.path.join('data', 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)

    # save as weights as ndarray
    np.save(os.path.join('data', 'glove', 'glove6b_init_%dd.npy' % emb_dim), weights)

