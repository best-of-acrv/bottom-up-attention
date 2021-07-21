# Process data

# vqa
python -m data_utils.create_dictionary
python -m data_utils.compute_softscore
python -m data_utils.detection_features_converter

# captioning
python -m data_utils.create_caption_input_data