import os

# construct the model for training
def get_model(args, dataset, pretrained=False):
    if args.task.lower() == 'captioning':
        from model.captioning_model import baseline

        model = baseline(args, dataset, pretrained=pretrained)
    elif args.task.lower() == 'vqa':
        from model.vqa_model import baseline

        model = baseline(args, dataset, pretrained=pretrained)
        model.w_emb.init_embedding(os.path.join(args.data_directory, 'glove', 'glove6b_init_300d.npy'))

    return model