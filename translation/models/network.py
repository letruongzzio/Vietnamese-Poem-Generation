import os
import sys
import random
import math
import torch
import torch.nn as nn

PROJECT_PATH = os.path.expanduser("~/Vietnamese-Poem-Generation/")
sys.path.append(PROJECT_PATH)
from constants import BOS_IDX, PAD_IDX


# GRU-based Seq2Seq model
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



# Transformer-based Seq2Seq model
class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the Transformer model.
    """
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        # `den`'s formula: 1 / (10000 ^ (2i / emb_size))
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size) # [emb_size//2]
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # [maxlen, 1]
        pos_embedding = torch.zeros((maxlen, emb_size)) # [maxlen, emb_size]
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # Add batch dimension, [1, maxlen, emb_size]

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding) # Register as buffer to move to device with the model

    def forward(self, token_embedding: torch.Tensor):
        # token_embedding: [batch_size, seq_len, emb_size]
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

# Embedding Layer
class TokenEmbedding(nn.Module):
    """
    Token Embedding for the Transformer model.
    """
    def __init__(self, vocab_size: int, emb_size: int, pretrained_embedding=None, freeze_embedding=False):
        super(TokenEmbedding, self).__init__()
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_embedding, padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=PAD_IDX) # padding_idx=0 for <pad> helps to avoid gradient explosion
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        # tokens: [batch_size, seq_len]
        # return: [batch_size, seq_len, emb_size]
        return self.embedding(tokens) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    """
    Seq2Seq Transformer model.

    Args:
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        emb_size (int): Embedding size.
        nhead (int): Number of attention heads.
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        dim_feedforward (int): Feedforward dimension.
        dropout (float): Dropout rate.
        device (torch.device): Device to run the model on.

    Returns:
        None
    """
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda"),
                 pretrained_embedding=None,
                 freeze_embedding=False):
        super(Seq2SeqTransformer, self).__init__()
        self.emb_size = emb_size
        self.device = device

        # Embedding + Positional Encoding
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size, pretrained_embedding, freeze_embedding)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size, pretrained_embedding, freeze_embedding)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

        # Transformer Model
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)

        # Linear layer to convert the output of the transformer to the output vocabulary size
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_pad_mask=None, tgt_pad_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Create a mask to prevent the model from attending to the padding tokens
        seq_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(self.device)

        if src_pad_mask is None:
            src_pad_mask = (src == PAD_IDX)
        if tgt_pad_mask is None:
            tgt_pad_mask = (tgt == PAD_IDX)

        src_pad_mask = src_pad_mask.to(self.device)

        outs = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask,
                                src_key_padding_mask=src_pad_mask.to(self.device),
                                tgt_key_padding_mask=tgt_pad_mask.to(self.device))

        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_pad_mask=None):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_key_padding_mask=src_pad_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)
    
