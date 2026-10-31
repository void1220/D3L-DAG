"""CVAE-Opt neural layers under GPL-3.0; see NOTICE."""
import torch
from torch import nn
from torch.nn import functional as F

class Embedding(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Embedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)

    def forward(self, input_data):
        output_data = self.embed(input_data)
        return output_data

class Encoder(nn.Module):

    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, update_fn, search_space_size, hidden_size):
        super(Encoder, self).__init__()
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.gru_decoder = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, search_space_size)
        self.fc2 = nn.Linear(hidden_size, search_space_size)
        self.rnn = rnn
        self.update_fn = update_fn

    def forward(self, instance, solution, instance_hidden, config):
        batch_size, sequence_size, input_size = instance.size()
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach()
        last_hh = None
        last_hh_2 = None
        reference_hidden = self.reference_embedding(reference_input)
        for j in range(1, solution.shape[1]):
            rnn_out, last_hh = self.rnn(reference_hidden, last_hh)
            enc_attn = self.encoder_attn(instance_hidden, rnn_out)
            context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)
            ptr = solution.t()[j].long()
            if self.update_fn is not None:
                instance = self.update_fn(instance, ptr.data)
                instance_hidden = self.instance_embedding(instance)
            reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            reference_hidden = self.reference_embedding(reference_input)
            rnn_input = torch.cat((reference_hidden, context), dim=2)
            rnn_out_2, last_hh_2 = self.gru_decoder(rnn_input, last_hh_2)
        mu = self.fc1(last_hh_2.squeeze(0))
        log_var = self.fc2(last_hh_2.squeeze(0))
        return (self.reparameterise(mu, log_var), mu, log_var)

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, 1 * hidden_size), requires_grad=True))

    def forward(self, instance_hidden, rnn_out):
        batch_size, _, hidden_size = instance_hidden.size()
        hidden = rnn_out.expand_as(instance_hidden)
        hidden = torch.cat((instance_hidden, hidden), 2)
        v = self.v.expand(batch_size, -1, -1)
        W = self.W.expand(batch_size, -1, -1)
        ret = torch.bmm(hidden, W)
        attns = torch.bmm(torch.relu(ret), v)
        attns = F.softmax(attns, dim=1)
        return attns

class Pointer(nn.Module):

    def __init__(self, encoder_attn, rnn, hidden_size, search_space_size):
        super(Pointer, self).__init__()
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, hidden_size), requires_grad=True))
        self.fc1 = nn.Linear(2 * hidden_size + search_space_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.encoder_attn = encoder_attn
        self.rnn = rnn

    def forward(self, instance_hidden, reference_hidden, Z, last_hh):
        rnn_out, last_hh = self.rnn(reference_hidden, last_hh)
        rnn_out = rnn_out
        enc_attn = self.encoder_attn(instance_hidden, rnn_out)
        context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)
        fc_input = torch.cat((context.squeeze(1), Z, reference_hidden.squeeze(1)), dim=1)
        fc_output = self.fc1(fc_input)
        fc_output = self.fc2(fc_output).unsqueeze(1)
        fc_output = fc_output.expand(-1, instance_hidden.size(1), -1)
        fc_output = torch.cat((instance_hidden, fc_output), dim=2)
        v = self.v.expand(instance_hidden.size(0), -1, -1)
        W = self.W.expand(instance_hidden.size(0), -1, -1)
        probs = torch.bmm(torch.tanh(torch.bmm(fc_output, W)), v).squeeze(2)
        return (probs, last_hh)
