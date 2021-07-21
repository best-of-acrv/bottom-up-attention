import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from nlgeval import NLGEval

class Evaluator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sample_directory = args.save_directory
        self.batch_size = args.batch_size

    # sample images using specified snapshot vqa model
    def sample_vqa(self, model, dataset):
        # create dataloader using dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        # output directory
        output_directory = os.path.join(self.sample_directory, 'output', 'vqa')
        os.makedirs(output_directory, exist_ok=True)

        # set model to eval mode for inference
        model.eval()
        # sample images and generate prediction images
        score = 0
        upper_bound = 0
        print('Evaluating Model...')
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # extract data and labels from batch
                features = batch['features']
                q_token = batch['q_token']
                answers = batch['target']
                image_ids = batch['image_id']
                questions = batch['question']

                if model.cuda_available:
                    features = features.cuda()
                    q_token = q_token.cuda()
                    answers = answers.cuda()

                # forward inference to compute logits
                logits = model(features, q_token)

                # get answer
                pred = torch.max(logits, dim=1)[1]

                batch_score = model.compute_score_with_logits(logits, answers).sum()
                score += batch_score
                upper_bound += (answers.max(dim=1)[0]).sum()

                # produce answers for each question
                # make sure number of predictions matches number of images
                num_batch_images = image_ids.shape[0]
                num_batch_questions = len(questions)
                num_batch_preds = pred.shape[0]
                assert num_batch_images == num_batch_preds == num_batch_questions

                for j in range(num_batch_images):
                    im_pred = pred[j]
                    image_id = image_ids[j]

                    answer = dataset.label2ans[im_pred]

                    # image filename
                    image_file = dataset.image_prefix + str(int(image_id)).zfill(12) + '.jpg'
                    im = Image.open(image_file)
                    # size of image
                    im_height = im.height
                    im_width = im.width

                    # pad image
                    pad_im = Image.new('RGB', (im_width, im_height+100))
                    pad_im.paste(im)

                    # write question and answer
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
                    draw = ImageDraw.Draw(pad_im)
                    draw.text((0, im_height + 30), questions[j], (255, 255, 255), font=font)
                    draw.text((0, im_height + 60), answer, (0, 255, 0), font=font)
                    draw = ImageDraw.Draw(pad_im)
                    pad_im.save(os.path.join(output_directory, str(int(image_id)).zfill(12) + '_pred.jpg'))

            score = score / len(dataloader.dataset)
            upper_bound = upper_bound / len(dataloader.dataset)

            with open(os.path.join(self.sample_directory, 'metrics_vqa.txt'), 'w') as f:
                f.writelines(['Score: ', str(score), '\n', 'Upper Bound: ', str(upper_bound), '\n'])


    # sample images using specified snapshot vqa model
    def sample_captioning(self, model, dataset):
        """
        Evaluation
        :param beam_size: beam size at which to generate captions for evaluation
        :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
        """
        # set beam size
        beam_size = 5

        # set vocab size
        rev_word_map = {v: k for k, v in dataset.word_map.items()}
        vocab_size = len(dataset.word_map)

        # output directory
        output_directory = os.path.join(self.sample_directory, 'output', 'captioning')
        os.makedirs(output_directory, exist_ok=True)

        # set model to eval mode
        model.eval()

        # load evaluator
        nlgeval = NLGEval()

        # DataLoader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=model.cuda_available)

        # Lists to store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        references = list()
        hypotheses = list()

        print('Evaluating Model...')
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                k = beam_size

                # extract data and labels from batch
                image_id = batch['imageid']
                features = batch['features']
                all_captions = batch['all_captions']

                # Move to device, if available
                if model.cuda_available:
                    features = features.cuda()
                    all_captions = all_captions.cuda()

                features_mean = features.mean(1)
                features_mean = features_mean.expand(k, 2048)

                # Tensor to store top k previous words at each step; now they're just <start>
                k_prev_words = torch.LongTensor([[dataset.word_map['<start>']]] * k)  # (k, 1)
                if model.cuda_available:
                    k_prev_words = k_prev_words.cuda()

                # Tensor to store top k sequences; now they're just <start>
                seqs = k_prev_words  # (k, 1)

                # Tensor to store top k sequences' scores; now they're just 0
                top_k_scores = torch.zeros_like(seqs)  # (k, 1)

                # Lists to store completed sequences and scores
                complete_seqs = list()
                complete_seqs_scores = list()

                # Start decoding
                step = 1
                h1, c1 = model.init_hidden_state(k)  # (batch_size, decoder_dim)
                h2, c2 = model.init_hidden_state(k)

                # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
                while True:
                    embeddings = model.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                    h1, c1 = model.top_down_attention(
                        torch.cat([h2, features_mean, embeddings], dim=1),
                        (h1, c1))  # (batch_size_t, decoder_dim)
                    attention_weighted_encoding = model.attention(features, h1)
                    h2, c2 = model.language_model(
                        torch.cat([attention_weighted_encoding, h1], dim=1), (h2, c2))

                    scores = model.fc(h2)  # (s, vocab_size)
                    scores = F.log_softmax(scores, dim=1)

                    # Add
                    scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                    # For the first step, all k points will have the same scores (since same k previous words, h, c)
                    if step == 1:
                        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                    else:
                        # Unroll and find top scores, and their unrolled indices
                        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                    # Convert unrolled indices to actual indices of scores
                    prev_word_inds = top_k_words // vocab_size  # (s)
                    next_word_inds = top_k_words % vocab_size  # (s)

                    # Add new words to sequences
                    seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                    # Which sequences are incomplete (didn't reach <end>)?
                    incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != dataset.word_map['<end>']]
                    complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                    # Set aside complete sequences
                    if len(complete_inds) > 0:
                        complete_seqs.extend(seqs[complete_inds].tolist())
                        complete_seqs_scores.extend(top_k_scores[complete_inds])
                    k -= len(complete_inds)  # reduce beam length accordingly

                    # Proceed with incomplete sequences
                    if k == 0:
                        break
                    seqs = seqs[incomplete_inds]
                    h1 = h1[prev_word_inds[incomplete_inds]]
                    c1 = c1[prev_word_inds[incomplete_inds]]
                    h2 = h2[prev_word_inds[incomplete_inds]]
                    c2 = c2[prev_word_inds[incomplete_inds]]
                    features_mean = features_mean[prev_word_inds[incomplete_inds]]
                    top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                    k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                    # Break if things have been going on too long
                    if step > 50:
                        break
                    step += 1

                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]

                # References
                img_caps = all_captions[0].tolist()
                img_captions = list(
                    map(lambda c: [rev_word_map[w] for w in c if
                                   w not in {dataset.word_map['<start>'], dataset.word_map['<end>'], dataset.word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                img_caps = [' '.join(c) for c in img_captions]
                references.append(img_caps)

                # Hypotheses
                hypothesis = ([rev_word_map[w] for w in seq if w not in {dataset.word_map['<start>'], dataset.word_map['<end>'], dataset.word_map['<pad>']}])
                hypothesis = ' '.join(hypothesis)
                hypotheses.append(hypothesis)
                assert len(references) == len(hypotheses)

                # save image with hypothesis
                for j, id in enumerate(image_id):
                    image_file = os.path.join(dataset.image_prefix, id)
                    im = Image.open(image_file)
                    # size of image
                    im_height = im.height
                    im_width = im.width

                    # pad image
                    pad_im = Image.new('RGB', (im_width, im_height + 100))
                    pad_im.paste(im)

                    # write question and answer
                    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 18)
                    draw = ImageDraw.Draw(pad_im)
                    draw.text((0, im_height + 30), hypothesis, (255, 255, 255), font=font)
                    draw = ImageDraw.Draw(pad_im)
                    pad_im.save(os.path.join(output_directory, id + '_pred.jpg'))

            # Calculate scores
            print('Computing metrics...')
            references = list(map(list, zip(*references)))
            metrics_dict = nlgeval.compute_metrics(references, hypotheses)
            print(metrics_dict)

            with open(os.path.join(self.sample_directory, 'metrics_captioning.txt'), 'w') as f:
                f.writelines(['BLEU-4: ', str(metrics_dict['Bleu_4']), '\n',
                              'METEOR: ', str(metrics_dict['METEOR']), '\n',
                              'ROUGE-L: ', str(metrics_dict['ROUGE_L']), '\n',
                              'CIDEr: ', str(metrics_dict['CIDEr']), '\n'])


