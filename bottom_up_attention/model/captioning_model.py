from nltk.translate.bleu_score import corpus_bleu
import os
import timeit
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

from ..helpers import AverageMeter, accuracy
from .helpers import download_model, find_snapshot


class Attention(nn.Module):

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):

        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(
            features_dim,
            attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(
            decoder_dim,
            attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(
            nn.Linear(attention_dim,
                      1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):

        att1 = self.features_att(
            image_features)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(
            self.relu(att1 + att2.unsqueeze(1)))).squeeze(
                2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        attention_weighted_encoding = (image_features *
                                       alpha.unsqueeze(2)).sum(
                                           dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding


class DecoderWithAttention(nn.Module):

    def __init__(self,
                 attention_dim,
                 embed_dim,
                 decoder_dim,
                 vocab_size,
                 features_dim=2048,
                 dropout=0.5):

        super(DecoderWithAttention, self).__init__()

        # check for cuda availability
        self.cuda_available = True if torch.cuda.is_available() else False

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(features_dim, decoder_dim,
                                   attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(
            embed_dim + features_dim + decoder_dim, decoder_dim,
            bias=True)  # top down attention LSTMCell
        self.language_model = nn.LSTMCell(features_dim + decoder_dim,
                                          decoder_dim,
                                          bias=True)  # language model LSTMCell
        self.fc = weight_norm(nn.Linear(
            decoder_dim,
            vocab_size))  # linear layer to find scores over vocabulary
        self.init_weights(
        )  # initialize some layers with the uniform distribution

        # loss criterions
        self.criterion_ce = nn.CrossEntropyLoss()

        # optimiser
        self.optimiser = None

    def attach_optimiser(self, learning_rate):
        self.optimiser = torch.optim.Adamax(self.parameters(),
                                            lr=learning_rate)

    def init_weights(self):

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):

        h = torch.zeros(batch_size,
                        self.decoder_dim)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim)

        if self.cuda_available:
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features.mean(
            1)  # (batch_size, num_pixels, encoder_dim)
        if self.cuda_available:
            image_features_mean = image_features_mean.cuda()

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(
            dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(
            batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(
            batch_size)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)

        if self.cuda_available:
            predictions = predictions.cuda()

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1, c1 = self.top_down_attention(
                torch.cat([
                    h2[:batch_size_t], image_features_mean[:batch_size_t],
                    embeddings[:batch_size_t, t, :]
                ],
                          dim=1), (h1[:batch_size_t], c1[:batch_size_t]))
            attention_weighted_encoding = self.attention(
                image_features[:batch_size_t], h1[:batch_size_t])
            h2, c2 = self.language_model(
                torch.cat([
                    attention_weighted_encoding[:batch_size_t],
                    h1[:batch_size_t]
                ],
                          dim=1), (h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

    def fit(self, scores, caps_sorted, decode_lengths):

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths,
                                      batch_first=True)[0]
        targets = pack_padded_sequence(targets,
                                       decode_lengths,
                                       batch_first=True)[0]

        # Calculate loss
        loss = self.criterion_ce(scores, targets)

        # Back prop.
        self.optimiser.zero_grad()
        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.parameters()), 0.25)

        # Update weights
        self.optimiser.step()

        # compute top 5 classification
        top5 = accuracy(scores, targets, 5)

        return loss, top5

    def validate(self, dataset, batch_size):
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        start = timeit.default_timer()

        references = list(
        )  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        # Batches
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # extract data and labels from batch
                features = batch['features']
                caption = batch['caption']
                caption_len = batch['caption_len']
                all_captions = batch['all_captions']

                # Move to device, if available
                if self.cuda_available:
                    features = features.cuda()
                    caption = caption.cuda()
                    caption_len = caption_len.cuda()
                    all_captions = all_captions.cuda()

                # forward inference to compute logits
                scores, caps_sorted, decode_lengths, sort_ind = self.forward(
                    features, caption, caption_len)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores = pack_padded_sequence(scores,
                                              decode_lengths,
                                              batch_first=True)[0]
                targets = pack_padded_sequence(targets,
                                               decode_lengths,
                                               batch_first=True)[0]

                # Calculate loss
                loss = self.criterion_ce(scores, targets)

                # Keep track of metrics
                stop = timeit.default_timer()
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(stop - start)
                start = stop

                print(
                    'Validation: [{0}/{1}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                        i,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top5=top5accs))

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

                # References
                all_captions = all_captions[
                    sort_ind]  # because images were sorted in the decoder
                for j in range(all_captions.shape[0]):
                    img_caps = all_captions[j].tolist()
                    img_captions = list(
                        map(
                            lambda c: [
                                w for w in c if w not in {
                                    dataset.word_map['<start>'], dataset.
                                    word_map['<pad>']
                                }
                            ], img_caps))  # remove <start> and pads
                    references.append(img_captions)

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = list()
                for j, p in enumerate(preds):
                    temp_preds.append(
                        preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)
        bleu4 = round(bleu4, 4)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'
            .format(loss=losses, top5=top5accs, bleu=bleu4))
        return bleu4

    def save(self, global_iteration, log_directory):
        os.makedirs(os.path.join(log_directory, 'snapshots'), exist_ok=True)
        model = {
            'model': self.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'global_iteration': global_iteration
        }

        model_path = os.path.join(
            log_directory, 'snapshots',
            'model-{:06d}.pth.tar'.format(global_iteration))
        print('Creating Snapshot: ' + model_path)
        torch.save(model, model_path)

    def load(self, log_directory=None, snapshot_num=None, with_optim=True):
        snapshot_dir = os.path.join(log_directory, 'snapshots')
        model_name = find_snapshot(snapshot_dir, snapshot_num)

        map_location = None
        if not self.cuda_available:
            map_location = torch.device('cpu')

        if model_name is None:
            print(
                'Model not found: initialising using default PyTorch initialisation!'
            )
            # uses pytorch default initialisation
            return 0
        # load model if snapshot was found
        else:
            full_model = torch.load(os.path.join(snapshot_dir, model_name),
                                    map_location=map_location)
            print('Loading model from: ' +
                  os.path.join(snapshot_dir, model_name))
            self.load_state_dict(full_model['model'], strict=False)
            if with_optim:
                self.optimiser.load_state_dict(full_model['optimiser'])
                # move optimiser to cuda
                if self.cuda_available:
                    for state in self.optimiser.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
            curr_iteration = full_model['global_iteration']
            return curr_iteration


pretrained_urls = {
    'baseline-captioning':
        'https://cloudstor.aarnet.edu.au/plus/s/xDJpTq3digOjuNZ/download',
}


def baseline(args, dataset, pretrained=False):

    # initialise model
    model = DecoderWithAttention(args,
                                 attention_dim=dataset.v_dim,
                                 embed_dim=1024,
                                 decoder_dim=args.num_hid,
                                 vocab_size=len(dataset.word_map),
                                 dropout=0.5)

    # load model on device if available
    map_location = None
    if not model.cuda_available:
        map_location = torch.device('cpu')

    # download and load pretrained model
    if pretrained:
        key = 'baseline-captioning'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(
            key, url, map_location=map_location)['model'],
                              strict=False)
    else:
        key = 'untrained'

    # set model name
    model.name = key

    return model
