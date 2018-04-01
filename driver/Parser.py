from transition.State import *
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time

class TransitionBasedParser(object):
    def __init__(self, encoder, decoder, root_id, config, ac_size):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.root = root_id
        encoder_p = next(filter(lambda p: p.requires_grad, encoder.parameters()))
        self.use_cuda = encoder_p.is_cuda
        self.bucket = Variable(torch.zeros(self.config.train_batch_size, 1, self.config.lstm_hiddens * 2)).type(torch.FloatTensor)
        self.cut = Variable(torch.zeros(self.config.train_batch_size, ac_size)).type(torch.FloatTensor)
        self.index = Variable(torch.zeros(self.config.train_batch_size * 4)).type(torch.LongTensor)
        self.device = encoder_p.get_device() if self.use_cuda else None
        if self.use_cuda:
            self.bucket = self.bucket.cuda(self.device)
            self.index = self.index.cuda(self.device)
            self.cut = self.cut.cuda(self.device)
        self.gold_pred_pairs = []
        self.training = True
        if self.config.train_batch_size > self.config.test_batch_size:
            batch_size = self.config.train_batch_size
        else:
            batch_size = self.config.test_batch_size
        self.batch_states = []
        self.step = []
        for idx in range(0, batch_size):
            self.batch_states.append([])
            self.step.append(0)
            for idy in range(0, 1024):
                self.batch_states[idx].append(State())

    def encode(self, words, extwords, tags, masks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device),
            tags = tags.cuda(self.device)
            masks = masks.cuda(self.device)
        self.encoder_outputs = self.encoder.forward(words, extwords, tags, masks)

    def compute_loss(self, true_acs):
        b, l1, l2 = self.decoder_outputs.size()
        true_acs = _model_var(
            self.encoder,
            pad_sequence(true_acs, length=l1, padding=-1, dtype=np.int64))
        arc_loss = F.cross_entropy(
            self.decoder_outputs.view(b * l1, l2), true_acs.view(b * l1),
            ignore_index=-1)
        return arc_loss

    def compute_accuracy(self):
        total_num = 0
        correct = 0
        for iter in self.gold_pred_pairs:
            gold_len = len(iter[0])
            pred_len = len(iter[1])
            assert gold_len == pred_len
            total_num += gold_len
            for idx in range(0, gold_len):
                if iter[0][idx] == iter[1][idx]:
                    correct += 1
        return total_num, correct


    def decode(self, batch_data, batch_step_actions, vocab):
        decoder_scores = []
        self.gold_pred_pairs.clear()

        self.b, self.l1, self.l2 = self.encoder_outputs.size()
        if self.b != self.bucket.size()[0]:
            self.bucket = Variable(torch.zeros(self.b, 1, self.l2)).type(torch.FloatTensor)
            if self.use_cuda:
                self.bucket = self.bucket.cuda(self.device)
        self.encoder_outputs = torch.cat((self.encoder_outputs, self.bucket), 1)
        for idx in range(0, self.b):
            start_state = self.batch_states[idx][0]
            start_state.clear()
            start_state.ready(batch_data[idx], vocab)
            self.step[idx] = 0
        global_step = 0
        while not self.all_states_are_finished():
            hidden_states = self.batch_prepare()
            if self.training:
                gold_actions = batch_step_actions[global_step]
            self.get_cut(vocab)
            action_scores = self.decoder.forward(hidden_states, self.cut)
            pred_ac_ids = self.get_predicted_ac_id(action_scores)
            pred_actions = self.get_predict_actions(pred_ac_ids, vocab)
            if self.training:
                self.move(gold_actions, vocab)
                self.gold_pred_pairs.append((gold_actions, pred_actions))
                decoder_scores.append(action_scores.unsqueeze(1))
            else:
                self.move(pred_actions, vocab)
            global_step += 1
        if self.training:
            self.decoder_outputs = torch.cat(decoder_scores, 1)

    def get_gold_actions(self, batch_gold_actions):
        gold_actions = []
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                gold_ac = batch_gold_actions[idx][cur_step]
                gold_actions.append(gold_ac)
        return gold_actions

    def get_predict_actions(self, pred_ac_ids, vocab):
        pred_actions = []
        for ac_id in pred_ac_ids:
            pred_ac = vocab.id2ac(ac_id)
            pred_actions.append(pred_ac)
        return pred_actions

    def all_states_are_finished(self):
        is_finish = True
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            if not cur_states[self.step[idx]].is_end():
                is_finish = False
                break
        return is_finish

    def batch_prepare(self):
        if self.b != self.index.size()[0]:
            self.index = Variable(torch.zeros(self.b * 4)).type(torch.LongTensor)
            index_data = np.array([0] * self.b * 4)
            if self.use_cuda:
                self.index = self.index.cuda(self.device)
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                s0, s1, s2, q0 = cur_states[cur_step].prepare_index(self.l1)
                offset_x = idx * 4
                offset_y = idx * (self.l1 + 1)
                index_data[offset_x + 0] = s0 + offset_y
                index_data[offset_x + 1] = s1 + offset_y
                index_data[offset_x + 2] = s2 + offset_y
                index_data[offset_x + 3] = q0 + offset_y
        self.index.data.copy_(torch.from_numpy(index_data))
        h_s = torch.index_select(self.encoder_outputs.view(self.b * (self.l1 + 1), self.l2), 0, self.index)
        h_s = h_s.view(self.b, 4 * self.l2)
        return h_s

    def get_predicted_ac_id(self, action_scores):
        action_scores = action_scores.data.cpu().numpy()
        ac_ids = []
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            if not cur_states[self.step[idx]].is_end():
                ac_id = np.argmax(action_scores[idx])
                ac_ids.append(ac_id)
        return ac_ids

    def move(self, pred_actions, vocab):
        #count = 0
        #for idx in range(0, self.b):
            #cur_states = self.batch_states[idx]
            #if not cur_states[self.step[idx]].is_end():
                #count += 1
        #assert len(pred_actions) == count
        offset = 0
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                next_state = self.batch_states[idx][cur_step + 1]
                cur_states[cur_step].move(next_state, pred_actions[offset])
                offset += 1
                self.step[idx] += 1


    def get_cut(self, vocab):
        all_mask = np.array([[False] * vocab.ac_size] * self.b)
        for idx in range(0, self.b):
            cur_states = self.batch_states[idx]
            cur_step = self.step[idx]
            if not cur_states[cur_step].is_end():
                mask = cur_states[cur_step].get_candidate_actions(vocab)
                all_mask[idx] = mask
        if self.b != self.cut.size()[0]:
            self.cut = Variable(torch.zeros(self.b, vocab.ac_size)).type(torch.FloatTensor)
            if self.use_cuda:
                self.cut = self.cut.cuda(self.device)
        self.cut.data.copy_(torch.from_numpy(all_mask * -1e+20))

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)
