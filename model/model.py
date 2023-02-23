from __future__ import division, print_function, unicode_literals 

#most of the code is python2 the above line allows us to use features from python3

import json
import math
import operator
import os
import random
from io import open
from Queue import PriorityQueue

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim

import policy

SOS_token = 0 #stands for start of sequence in NLP frameworks

EOS_token = 1 #stands for end of sequence
UNK_token = 2 #stands for unknown tokens
PAD_token = 3 #its a special token that is used to pad sequences of tokens to a fixed length.


# Shawn beam search decoding

'''
Given a trained model, we obtain a set of highly probable sequences. In practice, this problem is often intractable due to the size of Ω, 
which grows exponentially in sequence length. As a result, we resort to approximating the optimization problem using a decoding algorithm
that returns a set of k sequences F(pθ; γ), where F denotes the decoding algorithm, and γ denotes its hyper-parameters. Concretely, we consider two
decoding approaches: a deterministic decoding algorithm that produces a set of sequences using beam search with beam-width k, and a stochastic
decoding algorithm that forms a set of sequences using ancestral sampling until k unique sequences are obtained.1 We refer readers to Welleck et al.
(2020a) for detailed descriptions of those decoding algorithms. 
'''
class BeamSearchNode(object):  #class number 1 and this is the decoder
    def __init__(self, h, prevNode, wordid, logp, leng):
        self.h = h
        self.prevNode = prevNode
        self.wordid = wordid
        self.logp = logp
        self.leng = leng

    def eval(self, repeatPenalty, tokenReward, scoreTable, alpha=1.0):
        reward = 0
        alpha = 1.0

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

'''
The Long Short-Term Memory
(LSTM) is a specific RNN (recurrent neural network) architecture whose
design makes it much easier to train

its comparable to GRU 

 Gated Recurrent Unit (GRU) is a type of Recurrent Neural Network (RNN) 
 that, in certain cases, has advantages over long short term memory (LSTM). 
 GRU uses less memory and is faster than LSTM, however, LSTM is more accurate when using datasets with longer sequences.
 
 #so the question here is what is this sequence
'''
def init_lstm(cell, gain=1): #a method
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(1.0)
        hh_b[l // 4:l // 2].data.fill_(1.0)
'''
Chat GPT description of the above snippet of code:
The code snippet you provided is setting a positive bias for the forget gate in the weights of a Long Short-Term Memory (LSTM) cell. 
This technique was introduced in a paper by Jozefowicz et al. in 2015 and is known as "positive forget gate bias".

In an LSTM cell, the forget gate determines how much of the previous cell state should be forgotten or retained.
The positive forget gate bias means that the forget gate bias is set to a positive value, which biases the gate towards 
retaining more of the previous cell state. This can help to prevent the cell from forgetting important information that was stored in the previous time step.

The code sets the bias for the input-hidden and hidden-hidden weight matrices of the LSTM cell by filling a slice 
of the weights with the value 1.0. Specifically, it sets the bias for the forget gate weights to 1.0 for the input-hidden 
and hidden-hidden weight matrices. The slice of the weight matrix that is being modified is determined by the length of the 
input-hidden bias weights (len(ih_b)) and the fact that the LSTM weights are structured into four parts (input gate, forget gate, output gate, and cell input).
The slice for the forget gate weights corresponds to the second quarter of the weights (from l // 4 to l // 2).
'''




'''
The code you provided is a function for initializing the weights of a Gated Recurrent Unit (GRU) neural network layer in PyTorch.

The function takes a GRU layer as input and an optional gain parameter, which defaults to 1. 
It first resets the parameters of the GRU layer to their default values. 
Then, it iterates over the weights of the GRU layer and initializes the hidden-to-hidden weight matrix 
using the orthogonal initialization method (torch.nn.init.orthogonal_()).

The hidden-to-hidden weight matrix is divided into two parts: the reset gate and the update gate. 
Each part has a size of gru.hidden_size, which is the number of hidden units in the GRU layer. 
The loop iterates over each reset/update gate weight matrix in steps of gru.hidden_size 
(i.e., over each row of the hidden-to-hidden weight matrix), and initializes each gate weight matrix using orthogonal initialization.

Orthogonal initialization initializes the weight matrix with a random orthogonal matrix, 
which helps to prevent the gradients from exploding or vanishing during training. 
The gain parameter scales the orthogonal matrix by the given gain factor, which can help to improve the performance of the GRU layer.

Overall, this function initializes the weights of a GRU layer in a way that helps to prevent the gradients from vanishing or exploding, 
which can lead to better training and improved performance.




'''

def init_gru(gru, gain=1): #method 2
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i+gru.hidden_size],gain=gain)

'''
this is the definitionn structure

and nn refers to the below

import torch.nn as nn 

and is a neural network



the bidirectional parameter is also useful

The bidirectional parameter is an optional parameter in many PyTorch recurrent neural network (RNN) layers,
such as LSTM and GRU. When set to True, it enables the RNN layer to process the input sequence in both forward and backward directions.

By default, an RNN layer processes the input sequence in a single direction, either from the first 
time step to the last time step (i.e., forward direction) or from the last time step to the first time step (
i.e., backward direction). However, in many applications, information from both directions can be useful for 
predicting the output sequence. For example, in natural language processing, the meaning of a word in a sentence 
can depend on the words that come both before and after it.

By setting the bidirectional parameter to True, the RNN layer processes the input sequence in both directions, 
by creating two separate hidden states for each time step, one for the forward direction and one for the backward direction. 
The outputs from the forward and backward directions are then concatenated to produce the final output for each time step. 
This can improve the performance of the RNN layer by allowing it to capture both past and future context of the input sequence. 
However, it also increases the number of parameters and computation required by the RNN layer.
'''
def whatCellType(input_size, hidden_size, cell_type, dropout_rate): #method 3 that uses the previous methods
    if cell_type == 'rnn':
        cell = nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'gru':
        cell = nn.GRU(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'lstm':
        cell = nn.LSTM(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell
    elif cell_type == 'bigru':
        cell = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'bilstm':
        cell = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell

'''
The above code defines an encoder recurrent neural network (RNN) module in PyTorch.

The EncoderRNN class inherits from the nn.Module class and overrides its __init__ and forward methods.

In the __init__ method, the class initializes the encoder RNN by specifying the input size, 
embedding size, hidden size, cell type (e.g., GRU, LSTM), depth (number of layers), and dropout rate. 
It then creates an embedding layer to convert the input sequences into dense vector representations, 
and a recurrent layer using the specified cell type.

In the forward method, the encoder takes as input a batch of input sequences (input_seqs) of variable length, 
along with the lengths of each sequence (input_lens). The input sequences are first sorted by length in decreasing order, 
and then embedded using the embedding layer. The resulting embeddings are packed into a PackedSequence object and fed 
into the recurrent layer. The outputs and hidden states of the recurrent layer are then returned, with 
the outputs optionally processed to remove the effect of bidirectionality.

The purpose of the encoder RNN is to encode the input sequence into a fixed-length vector 
representation that can be used by a decoder RNN to generate the output sequence. 
The encoder achieves this by processing the input sequence one element at a time, 
and updating its internal hidden state at each time step based on the current input 
and the previous hidden state. The final hidden state of the encoder represents a summary 
of the entire input sequence, and is used as the initial hidden state of the decoder.
'''
class EncoderRNN(nn.Module): #class 2 and this is the encoder
    def __init__(self, input_size,  embedding_size, hidden_size, cell_type, depth, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.n_layers = depth
        self.dropout = dropout
        self.bidirectional = False
        if 'bi' in cell_type:
            self.bidirectional = True
        padding_idx = 3
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.rnn = whatCellType(embedding_size, hidden_size,
                    cell_type, dropout_rate=self.dropout)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        input_lens = np.asarray(input_lens)
        input_seqs = input_seqs.transpose(0,1)
        #batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx))
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()

        if isinstance(hidden, tuple):
            hidden = list(hidden)
            hidden[0] = hidden[0].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
            hidden[1] = hidden[1].transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
            hidden = tuple(hidden)
        else:
            hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()

        return outputs, hidden
'''
A sequence model:

A sequence-to-sequence (Seq2Seq) model is a type of neural network architecture designed for handling sequences of different lengths,
and is often used for tasks such as machine translation, speech recognition, and text summarization. 
The Seq2Seq model consists of two parts: an encoder network that processes the input sequence and converts
it into a fixed-length vector representation, and a decoder network that uses the encoded vector to generate an output sequence.

The encoder network can be any type of recurrent neural network (RNN), 
such as a long short-term memory (LSTM) or gated recurrent unit (GRU), 
and its output is a vector that summarizes the input sequence.
This vector is then fed into the decoder network, which is typically another RNN, 
that generates the output sequence one element at a time.
At each time step, 
the decoder uses the previously generated output (or a special start-of-sequence token at the first step)
and the encoded input vector to produce the next output element. 
The training objective is to minimize the difference between the generated output sequence and the target output sequence,
often using a loss function like cross-entropy.




Based off the previous statements like the above everything feels pretty standard for this model except for the decoder whihc is a beam
search node decoder. We are trying to go and check which is teh best type of model. I dont think we have got to the loss function yet
'''

'''
This is a PyTorch implementation of an Attention mechanism for use in a sequence-to-sequence model.

The class Attn takes in two arguments, method and hidden_size, where method is a string specifying the type of attention to use 
(e.g. "dot", "general", "concat"), and hidden_size is the size of the hidden state in the encoder and decoder.

The forward method takes in two inputs, hidden and encoder_outputs, 
where hidden is the previous hidden state of the decoder and encoder_outputs are the encoder outputs from the Encoder. 
The method then computes the attention score between the hidden state and the encoder_outputs using the score method.

The score method concatenates the hidden state and encoder_outputs, 
applies a linear transformation (attn), applies a hyperbolic tangent function (tanh) 
and a dot product between the result and a learnable parameter v (self.v). 
The final result is then returned as the attention energies.

'''
'''
The attention mechanism in sequence-to-sequence (seq2seq) models is a technique that helps the decoder 
focus on specific parts of the input sequence when generating each output token.

In a traditional seq2seq model, the entire input sequence is encoded into a fixed-length vector, 
which is then fed into the decoder to generate the output sequence. However, 
this approach can be limiting because the fixed-length vector has to capture all the information in the input sequence,
and may not be able to handle longer sequences or more complex relationships between the input and output.

The attention mechanism addresses this issue by allowing the decoder to selectively focus on different parts of the input sequence at each time step. 
Specifically, at each step of decoding, the attention mechanism computes a set of weights that indicate how much attention to give to each input element.
These weights are then used to compute a weighted sum of the input elements, 
which is combined with the current hidden state of the decoder to generate the output token.

By allowing the decoder to selectively focus on different parts of the input sequence, 
the attention mechanism can improve the accuracy of seq2seq models, 
particularly for longer sequences or more complex relationships between the input and output.

'''
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)

        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)  # [T,B,H] -> [B,T,H]
        attn_energies = self.score(H,encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        cat = torch.cat([hidden, encoder_outputs], 2)
        energy = torch.tanh(self.attn(cat)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

'''
This is a decoder module in a sequence-to-sequence model that uses attention mechanism. 
It takes as input the target token (in the form of an index) and the previous hidden state of the decoder, 
and returns the output probability distribution over the target vocabulary and the new hidden state of the decoder.

The attention mechanism is used to compute a context vector that summarizes the relevant parts of the encoder's outputs 
for predicting the next target token. It does this by computing a score for each encoder output based on how well it 
matches the current decoder hidden state, and then using these scores to compute a weighted sum of the encoder outputs,
where the weights are given by the softmax of the scores.

The attention weights and the context vector are computed using the following steps:

Concatenate the current decoder hidden state with the encoder outputs along the time dimension
Apply a linear layer to this concatenated tensor to obtain a score for each encoder output
Apply the softmax function to these scores to obtain the attention weights
Compute the weighted sum of the encoder outputs using these attention weights to obtain the context vector.
The decoder then concatenates the context vector with the embedded input token, and feeds it to an RNN 
(whose type is specified by the cell_type argument) along with the previous decoder hidden state, 
to obtain the new decoder hidden state and the output probability distribution over the target vocabulary. 
The output probability distribution is obtained by applying a linear layer followed by a log-softmax activation function to the output of the RNN.
'''
class SeqAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout_p=0.1, max_length=30):
        super(SeqAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.output_size = output_size
        self.n_layers = 1
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size + hidden_size, hidden_size, cell_type, dropout_rate=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        self.score = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, embedding_size)

        # attention
        self.method = 'concat'
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, input, hidden, encoder_outputs):
        if isinstance(hidden, tuple):
            h_t = hidden[0]
        else:
            h_t = hidden
        encoder_outputs = encoder_outputs.transpose(0, 1)
        embedded = self.embedding(input)  # .view(1, 1, -1)
        # embedded = F.dropout(embedded, self.dropout_p)

        # SCORE 3
        max_len = encoder_outputs.size(1)
        h_t = h_t.transpose(0, 1)  # [1,B,D] -> [B,1,D]
        h_t = h_t.repeat(1, max_len, 1)  # [B,1,D]  -> [B,T,D]
        energy = self.attn(torch.cat((h_t, encoder_outputs), 2))  # [B,T,2D] -> [B,T,D]
        energy = torch.tanh(energy)
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        attn_weights = F.softmax(energy, dim=2)  # [B,1,T]

        # getting context
        context = torch.bmm(attn_weights, encoder_outputs)  # [B,1,H]

        # context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) #[B,1,H]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2)
        rnn_input = rnn_input.transpose(0, 1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden  # , attn_weights


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        padding_idx = 3
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx
                                      )
        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size, hidden_size, cell_type, dropout_rate=dropout)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1,B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        output = embedded
        #output = F.relu(embedded)

        output, hidden = self.rnn(output, hidden)

        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)

        return output, hidden


class Model(nn.Module):
    def __init__(self, args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index):
        super(Model, self).__init__()
        self.args = args
        self.max_len = args.max_len

        self.output_lang_index2word = output_lang_index2word
        self.input_lang_index2word = input_lang_index2word

        self.output_lang_word2index = output_lang_word2index
        self.input_lang_word2index = input_lang_word2index

        self.hid_size_enc = args.hid_size_enc
        self.hid_size_dec = args.hid_size_dec
        self.hid_size_pol = args.hid_size_pol

        self.emb_size = args.emb_size
        self.db_size = args.db_size
        self.bs_size = args.bs_size
        self.cell_type = args.cell_type
        if 'bi' in self.cell_type:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.depth = args.depth
        self.use_attn = args.use_attn
        self.attn_type = args.attention_type

        self.dropout = args.dropout
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.teacher_forcing_ratio = args.teacher_ratio
        self.vocab_size = args.vocab_size
        self.epsln = 10E-5


        torch.manual_seed(args.seed)
        self.build_model()
        self.getCount()
        try:
            assert self.args.beam_width > 0
            self.beam_search = True
        except:
            self.beam_search = False

        self.global_step = 0

    def cuda_(self, var):
        return var.cuda() if self.args.cuda else var

    def build_model(self):
        self.encoder = EncoderRNN(len(self.input_lang_index2word), self.emb_size, self.hid_size_enc,
                                  self.cell_type, self.depth, self.dropout).to(self.device)

        self.policy = policy.DefaultPolicy(self.hid_size_pol, self.hid_size_enc, self.db_size, self.bs_size).to(self.device)

        if self.use_attn:
            if self.attn_type == 'bahdanau':
                self.decoder = SeqAttnDecoderRNN(self.emb_size, self.hid_size_dec, len(self.output_lang_index2word), self.cell_type, self.dropout, self.max_len).to(self.device)
        else:
            self.decoder = DecoderRNN(self.emb_size, self.hid_size_dec, len(self.output_lang_index2word), self.cell_type, self.dropout).to(self.device)

        if self.args.mode == 'train':
            self.gen_criterion = nn.NLLLoss(ignore_index=3, size_average=True)  # logsoftmax is done in decoder part
            self.setOptimizers()

    def train(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor, dial_name=None):
        proba, _, decoded_sent = self.forward(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor)

        proba = proba.view(-1, self.vocab_size)
        self.gen_loss = self.gen_criterion(proba, target_tensor.view(-1))

        self.loss = self.gen_loss
        self.loss.backward()
        grad = self.clipGradients()
        self.optimizer.step()
        self.optimizer.zero_grad()

        #self.printGrad()
        return self.loss.item(), 0, grad

    def setOptimizers(self):
        self.optimizer_policy = None
        if self.args.optim == 'sgd':
            self.optimizer = optim.SGD(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adadelta':
            self.optimizer = optim.Adadelta(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adam':
            self.optimizer = optim.Adam(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)

    def forward(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor):
        """Given the user sentence, user belief state and database pointer,
        encode the sentence, decide what policy vector construct and
        feed it as the first hiddent state to the decoder."""
        target_length = target_tensor.size(1)

        # for fixed encoding this is zero so it does not contribute
        batch_size, seq_len = input_tensor.size()

        # ENCODER
        encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)

        # POLICY
        decoder_hidden = self.policy(encoder_hidden, db_tensor, bs_tensor)

        # GENERATOR
        # Teacher forcing: Feed the target as the next input
        _, target_len = target_tensor.size()
        decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=self.device)

        proba = torch.zeros(batch_size, target_length, self.vocab_size)  # [B,T,V]

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            use_teacher_forcing = True if random.random() < self.args.teacher_ratio else False
            if use_teacher_forcing:
                decoder_input = target_tensor[:, t].view(-1, 1)  # [B,1] Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

            proba[:, t, :] = decoder_output

        decoded_sent = None

        return proba, None, decoded_sent

    def predict(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor):
        with torch.no_grad():
            # ENCODER
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, input_lengths)

            # POLICY
            decoder_hidden = self.policy(encoder_hidden, db_tensor, bs_tensor)

            # GENERATION
            decoded_words = self.decode(target_tensor, decoder_hidden, encoder_outputs)

        return decoded_words, 0

    def decode(self, target_tensor, decoder_hidden, encoder_outputs):
        decoder_hiddens = decoder_hidden

        if self.beam_search:  # wenqiang style - sequicity
            decoded_sentences = []
            for idx in range(target_tensor.size(0)):
                if isinstance(decoder_hiddens, tuple):  # LSTM case
                    decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
                else:
                    decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
                encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

                # Beam start
                self.topk = 1
                endnodes = []  # stored end nodes
                number_required = min((self.topk + 1), self.topk - len(endnodes))
                decoder_input = torch.LongTensor([[SOS_token]], device=self.device)

                # starting node hidden vector, prevNode, wordid, logp, leng,
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                nodes = PriorityQueue()  # start the queue
                nodes.put((-node.eval(None, None, None, None),
                           node))

                # start beam search
                qsize = 1
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000: break

                    # fetch the best node
                    score, n = nodes.get()
                    decoder_input = n.wordid
                    decoder_hidden = n.h

                    if n.wordid.item() == EOS_token and n.prevNode != None:  # its not empty
                        endnodes.append((score, n))
                        # if reach maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)

                    log_prob, indexes = torch.topk(decoder_output, self.args.beam_width)
                    nextnodes = []

                    for new_k in range(self.args.beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                        score = -node.eval(None, None, None, None)
                        nextnodes.append((score, node))

                    # put them into queue
                    for i in range(len(nextnodes)):
                        score, nn = nextnodes[i]
                        nodes.put((score, nn))

                    # increase qsize
                    qsize += len(nextnodes)

                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for n in range(self.topk)]

                utterances = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        n = n.prevNode
                        utterance.append(n.wordid)

                    utterance = utterance[::-1]
                    utterances.append(utterance)

                decoded_words = utterances[0]
                decoded_sentence = [self.output_index2word(str(ind.item())) for ind in decoded_words]
                #print(decoded_sentence)
                decoded_sentences.append(' '.join(decoded_sentence[1:-1]))

            return decoded_sentences

        else:  # GREEDY DECODING
            decoded_sentences = self.greedy_decode(decoder_hidden, encoder_outputs, target_tensor)
            return decoded_sentences

    def greedy_decode(self, decoder_hidden, encoder_outputs, target_tensor):
        decoded_sentences = []
        batch_size, seq_len = target_tensor.size()
        decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=self.device)

        decoded_words = torch.zeros((batch_size, self.max_len))
        for t in range(self.max_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # get candidates
            topi = topi.view(-1)

            decoded_words[:, t] = topi
            decoder_input = topi.detach().view(-1, 1)

        for sentence in decoded_words:
            sent = []
            for ind in sentence:
                if self.output_index2word(str(int(ind.item()))) == self.output_index2word(str(EOS_token)):
                    break
                sent.append(self.output_index2word(str(int(ind.item()))))
            decoded_sentences.append(' '.join(sent))

        return decoded_sentences

    def clipGradients(self):
        grad = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
        return grad

    def saveModel(self, iter):
        print('Saving parameters..')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.encoder.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.enc')
        torch.save(self.policy.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.pol')
        torch.save(self.decoder.state_dict(), self.model_dir + self.model_name + '-' + str(iter) + '.dec')

        with open(self.model_dir + self.model_name + '.config', 'w') as f:
            f.write(unicode(json.dumps(vars(self.args), ensure_ascii=False, indent=4)))

    def loadModel(self, iter=0):
        print('Loading parameters of iter %s ' % iter)
        self.encoder.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.enc'))
        self.policy.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.pol'))
        self.decoder.load_state_dict(torch.load(self.model_dir + self.model_name + '-' + str(iter) + '.dec'))

    def input_index2word(self, index):
        if self.input_lang_index2word.has_key(index):
            return self.input_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def output_index2word(self, index):
        if self.output_lang_index2word.has_key(index):
            return self.output_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def input_word2index(self, index):
        if self.input_lang_word2index.has_key(index):
            return self.input_lang_word2index[index]
        else:
            return 2

    def output_word2index(self, index):
        if self.output_lang_word2index.has_key(index):
            return self.output_lang_word2index[index]
        else:
            return 2

    def getCount(self):
        learnable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_cnt = sum([reduce((lambda x, y: x * y), param.shape) for param in learnable_parameters])
        print('Model has', param_cnt, ' parameters.')

    def printGrad(self):
        learnable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        for idx, param in enumerate(learnable_parameters):
            print(param.grad, param.shape)
