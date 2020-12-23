import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers.download_helper import download_model
from helpers.model_helper import find_snapshot
from model.attention import Attention, NewAttention
from model.language_model import WordEmbedding, QuestionEmbedding
from model.classifier import SimpleClassifier
from model.fc import FCNet

class BaseModel(nn.Module):
    def __init__(self, args, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()

        # check for cuda availability
        self.cuda_available = True if torch.cuda.is_available() else False
        self.batch_size = args.batch_size

        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

        # optimiser
        self.optimiser = torch.optim.Adamax(self.parameters(), lr=args.learning_rate)

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        # [batch, q_dim]
        q_emb = self.q_emb(w_emb)

        att = self.v_att(v, q_emb)
        # [batch, v_dim]
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits

    def fit(self, output, answer, criterion):
        loss = criterion(output, answer)

        # do a backward pass to compute gradients
        self.optimiser.zero_grad()
        loss.backward()

        # gradient clipping helps training stability
        nn.utils.clip_grad_norm(self.parameters(), 0.25)

        # do optimiser step
        self.optimiser.step()

        return loss

    def compute_score_with_logits(self, logits, labels):

        # argmax on logits
        logits = torch.max(logits, dim=1)[1]
        one_hots = torch.zeros_like(labels)
        if self.cuda_available:
            one_hots = one_hots.cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots*labels

        return scores

    def validate(self, dataset):
        score = 0
        upper_bound = 0
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        with torch.no_grad():
            for batch in dataloader:
                # extract data and labels from batch
                features = batch['features']
                q_token = batch['q_token']
                answer = batch['target']

                if self.cuda_available:
                    features = features.cuda()
                    q_token = q_token.cuda()
                    answer = answer.cuda()

                # compute logits
                output = self.forward(features, q_token)
                batch_score = self.compute_score_with_logits(output, answer).sum()
                score += batch_score
                upper_bound += (answer.max(dim=1)[0]).sum()

            score = score / len(dataloader.dataset)
            upper_bound = upper_bound / len(dataloader.dataset)
        return score, upper_bound

    def save(self, global_iteration, log_directory):
        os.makedirs(os.path.join(log_directory, 'snapshots'), exist_ok=True)
        model = {
            'model': self.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'global_iteration': global_iteration
        }

        model_path = os.path.join(log_directory, 'snapshots', 'model-{:06d}.pth.tar'.format(global_iteration))
        print('Creating Snapshot: ' + model_path)
        torch.save(model, model_path)

    def load(self, log_directory=None, snapshot_num=None, with_optim=True):
        snapshot_dir = os.path.join(log_directory, 'snapshots')
        model_name = find_snapshot(snapshot_dir, snapshot_num)

        map_location = None
        if not self.cuda_available:
            map_location = torch.device('cpu')

        if model_name is None:
            print('Model not found: initialising using default PyTorch initialisation!')
            # uses pytorch default initialisation
            return 0
        # load model if snapshot was found
        else:
            full_model = torch.load(os.path.join(snapshot_dir, model_name), map_location=map_location)
            print('Loading model from: ' + os.path.join(snapshot_dir, model_name))
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
    'baseline-vqa': 'https://cloudstor.aarnet.edu.au/plus/s/a0b8IH8h0ZnCvAZ/download',
}

def baseline(args, dataset, pretrained=False):

    # initialise model
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, args.num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, args.num_hid)
    q_net = FCNet([args.num_hid, args.num_hid])
    v_net = FCNet([dataset.v_dim, args.num_hid])
    classifier = SimpleClassifier(args.num_hid, 2*args.num_hid, dataset.num_ans_candidates, 0.5)
    model = BaseModel(args, w_emb, q_emb, v_att, q_net, v_net, classifier)

    # load model on device if available
    map_location = None
    if not model.cuda_available:
        map_location = torch.device('cpu')

    # download and load pretrained model
    if pretrained:
        key = 'baseline-vqa'
        url = pretrained_urls[key]
        model.load_state_dict(download_model(key, url, map_location=map_location)['model'], strict=False)
    else:
        key = 'untrained'

    # set model name
    model.name = key

    return model


