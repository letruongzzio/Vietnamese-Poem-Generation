import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_PATH = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_PATH)
from constants import BOS_IDX, PAD_IDX

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, pretrained_embedding=None, freeze_embedding=False):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_IDX)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor):
        embedded = self.dropout(self.embedding(input_tensor))
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, pretrained_embedding=None, freeze_embedding=False):
        super(DecoderGRU, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_IDX)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor, hidden):
        output = self.embedding(input_tensor)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return self.softmax(output), hidden

class Seq2Seq_GRU(nn.Module):
    def __init__(self, encoder, decoder, device, bos_idx=BOS_IDX, teacher_forcing_ratio=0.5):
        super(Seq2Seq_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.BOS_IDX = bos_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src_ids, tgt_ids, training=True):
        batch_size = tgt_ids.size(0)
        seq_len = tgt_ids.size(1)

        decoder_input = torch.full((batch_size, 1), self.BOS_IDX, dtype=torch.long, device=self.device)
        encoder_output, decoder_hidden = self.encoder(src_ids)
        decoder_outputs = []

        for i in range(seq_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # Scheduled Sampling
            use_teacher_forcing = random.random() < self.teacher_forcing_ratio if training else False
            if use_teacher_forcing:
                decoder_input = tgt_ids[:, i].unsqueeze(1)  # Teacher forcing
            else:
                decoder_input = decoder_output.argmax(dim=-1)  # Use its own predictions as the next input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs
