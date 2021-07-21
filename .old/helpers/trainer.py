import timeit
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers.ops import AverageMeter, adjust_learning_rate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()

        # create directory for outputs
        self.eval_interval = args.eval_interval
        self.display_interval = args.display_interval
        self.snapshot_interval = args.snapshot_interval
        self.load_directory = args.load_directory
        self.save_directory = args.save_directory

        # training parameters
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs

    def train_captioning(self, args, model, dataset):

        train_dataset = dataset['train']
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # start training
        model.train()
        start = timeit.default_timer()

        # keep track of values to log
        batch_time = AverageMeter()  # forward prop. + back prop. time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        # main training loop
        curr_epoch = 0
        curr_iteration = 0
        best_bleu4 = 0
        epochs_since_improvement = 0
        for epoch in range(self.max_epochs):

            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if epochs_since_improvement == 20:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
                adjust_learning_rate(model.optimiser, 0.8)

            # reset iteration for after each epoch
            epoch_iteration = 0
            for batch in dataloader:
                # extract data and labels from batch
                features = batch['features']
                caption = batch['caption']
                caption_len = batch['caption_len']

                # move to GPU if available
                if model.cuda_available:
                    features = features.cuda()
                    caption = caption.cuda()
                    caption_len = caption_len.cuda()

                # forward inference to compute logits
                scores, caps_sorted, decode_lengths, sort_ind = model(features, caption, caption_len)

                # fit to model
                loss, top5 = model.fit(scores, caps_sorted, decode_lengths)

                # Keep track of metrics
                stop = timeit.default_timer()
                losses.update(loss.item(), sum(decode_lengths))
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(stop - start)
                start = stop

                # display current training loss
                if (curr_iteration + 1) % self.display_interval == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Top-5 Accuracy: {top5.val:.3f} ({top5.avg:.3f})'.format(curr_epoch, epoch_iteration + 1, len(dataloader),
                                                                                   batch_time=batch_time,
                                                                                   loss=losses,
                                                                                   top5=top5accs))
                epoch_iteration += 1
                curr_iteration += 1

            # evaluate model on validation
            if epoch % self.eval_interval == 0 and dataset['val']:
                # set model to eval first
                model.eval()
                recent_bleu4 = model.validate(dataset['val'])
                model.train()

            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
            else:
                # save our best model after each epoch
                epochs_since_improvement = 0
                model.save(curr_epoch + 1, self.save_directory)

            # update current epoch
            curr_epoch += 1


    def train_vqa(self, args, model, dataset):

        # setup dataloader for VQA training set
        train_dataset = dataset['train']
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # make save directory
        os.makedirs(self.save_directory, exist_ok=True)

        # define loss criterion
        criterion = nn.BCEWithLogitsLoss()

        # start training
        model.train()
        start = timeit.default_timer()

        # main training loop
        curr_epoch = 0
        curr_iteration = 0
        best_eval_score = 0
        for epoch in range(self.max_epochs):

            # reset iteration for after each epoch
            epoch_iteration = 0
            for batch in dataloader:
                # extract data and labels from batch
                features = batch['features']
                q_token = batch['q_token']
                answer = batch['target']

                # move to GPU if available
                if model.cuda_available:
                    features = features.cuda()
                    q_token = q_token.cuda()
                    answer = answer.cuda()

                # forward inference to compute logits
                output = model(features, q_token)

                # compute loss and score
                mean_loss = model.fit(output, answer, criterion)
                batch_score = model.compute_score_with_logits(output, answer).sum()

                # Keep track of metrics
                stop = timeit.default_timer()

                # display current training loss
                if (curr_iteration + 1) % self.display_interval == 0:
                    stop = timeit.default_timer()

                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Batch Time: {3}\t'
                          'Loss: {4}\t'
                          'Score: {5})'.format(curr_epoch, epoch_iteration + 1, len(dataloader),
                                                                          stop-start,
                                                                          mean_loss.item(),
                                                                          batch_score))

                    # print('Epoch: [{0}][{1}/{2}]\t'
                    #       [Time: {:3f}] [Lr: {}] [loss: {:4f}] [score: {:4f}]'.format(curr_epoch,
                    #                                                                                           curr_iteration + 1,
                    #                                                                                           stop - start,
                    #                                                                                           get_lr(model.optimiser),
                    #                                                                                           mean_loss.item(),
                    #                                                                                           batch_score))

                start = stop

                epoch_iteration += 1
                curr_iteration += 1

            # evaluate model by computing mean IU
            if (epoch + 1) % self.eval_interval == 0 and dataset['val']:

                # set model to eval first
                model.eval()
                eval_score, bound = model.validate(dataset['val'])
                print('[Epoch: {}] [Iter: {}] [eval_score: {:4f}] [upper_bound: {:4f}]'.format(curr_epoch,
                                                                                               curr_iteration + 1,
                                                                                               eval_score*100,
                                                                                               bound*100))
                model.train()

            # save our best model after each epoch
            if eval_score > best_eval_score:
                model.save(curr_epoch + 1, self.save_directory)
                best_eval_score = eval_score

            # update current epoch
            curr_epoch += 1





