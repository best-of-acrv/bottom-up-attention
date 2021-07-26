import glob
import json
import os

from .vqa import Dictionary


def compute_softscore():
    pass


def convert_detection_features():
    pass


def create_caption_input_data():
    pass


def make_dictionary(output_file, data_dirs):
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


def create_glove_embeddings():
    pass
