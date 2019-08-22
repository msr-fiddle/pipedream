# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import translation.seq2seq.data.config as config
import pack_utils._C as C


class Revert_varlen(torch.autograd.Function):
   @staticmethod
   def forward(ctx, input, lengths):
      ctx.lengths = lengths
      return C.revert_varlen_tensor(input,lengths)

   @staticmethod
   def backward(ctx, grad_output):
       return C.revert_varlen_tensor(grad_output, ctx.lengths), None

revert_varlen = Revert_varlen.apply

class EmuBidirLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bias = True, batch_first = False, bidirectional = True):
        super(EmuBidirLSTM, self).__init__()
        assert num_layers == 1, "emulation bidirectional lstm works for a single layer only"
        assert batch_first == False, "emulation bidirectional lstm works for batch_first = False only"
        assert bidirectional == True, "use for bidirectional lstm only"
        self.bidir = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, bidirectional = True)
        self.layer1 = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first)
        self.layer2 = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first)
        self.layer1.weight_ih_l0 = self.bidir.weight_ih_l0
        self.layer1.weight_hh_l0 = self.bidir.weight_hh_l0
        self.layer2.weight_ih_l0 = self.bidir.weight_ih_l0_reverse
        self.layer2.weight_hh_l0 = self.bidir.weight_hh_l0_reverse
        self.layer1.bias_ih_l0 = self.bidir.bias_ih_l0
        self.layer1.bias_hh_l0 = self.bidir.bias_hh_l0
        self.layer2.bias_ih_l0 = self.bidir.bias_ih_l0_reverse
        self.layer2.bias_hh_l0 = self.bidir.bias_hh_l0_reverse

    @staticmethod
    def bidir_lstm(model, input, lengths):
        packed_input = pack_padded_sequence(input, lengths.cpu().numpy())
        out =  model(packed_input)[0]
        return pad_packed_sequence(out)[0]

    @staticmethod
    def emu_bidir_lstm(model0, model1, input, lengths):#mask):
        mask = C.set_mask_cpp(lengths).unsqueeze(-1).cuda(non_blocking = True).type_as(input)
        inputl1 = revert_varlen(input, lengths)
        out1 = model1(inputl1)
        outputs = revert_varlen(out1[0], lengths)
        out0 = model0(input)[0]*mask
        out_bi = torch.cat([out0, outputs], dim=2)
        return out_bi

    def forward(self, input, lengths):
        lengths = lengths.cpu().long()
        if (input.size(1) > 128):
            return self.bidir_lstm(self.bidir, input, lengths)
        else:
            return self.emu_bidir_lstm(self.layer1, self.layer2, input, lengths)






class ResidualRecurrentEncoder(nn.Module):
    """
    Encoder with Embedding, LSTM layers, residual connections and optional
    dropout.

    The first LSTM layer is bidirectional and uses variable sequence length API,
    the remaining (num_layers-1) layers are unidirectional. Residual
    connections are enabled after third LSTM layer, dropout is applied between
    LSTM layers.
    """
    def __init__(self, vocab_size, hidden_size=128, num_layers=8, bias=True,
                 dropout=0, batch_first=False, embedder=None):
        """
        Constructor for the ResidualRecurrentEncoder.

        :param vocab_size: size of vocabulary
        :param hidden_size: hidden size for LSTM layers
        :param num_layers: number of LSTM layers, 1st layer is bidirectional
        :param bias: enables bias in LSTM layers
        :param dropout: probability of dropout (between LSTM layers)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param embedder: embedding module, if None constructor will create new
            embedding layer
        """
        super(ResidualRecurrentEncoder, self).__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            EmuBidirLSTM(hidden_size, hidden_size, num_layers=1, bias=bias, batch_first=batch_first, bidirectional=True))
#            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias,
#                    batch_first=batch_first, bidirectional=True))

        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=bias,
                    batch_first=batch_first))

        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=bias,
                        batch_first=batch_first))

        self.dropout = nn.Dropout(p=dropout)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=config.PAD)

    def forward(self, inputs, lengths):
        """
        Execute the encoder.

        :param inputs: tensor with indices from the vocabulary
        :param lengths: vector with sequence lengths (excluding padding)

        returns: tensor with encoded sequences
        """
        x = self.embedder(inputs)

        # bidirectional layer
#        x = pack_padded_sequence(x, lengths.cpu().numpy(),
#                                 batch_first=self.batch_first)
#        x, _ = self.rnn_layers[0](x)
        x = self.rnn_layers[0](x, lengths.cpu().long())
#        x, _ = pad_packed_sequence(x, batch_first=self.batch_first)

        # 1st unidirectional layer
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x)

        # the rest of unidirectional layers,
        # with residual connections starting from 3rd layer
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x)
            x = x + residual

        return x
