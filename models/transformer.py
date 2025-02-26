import os
import sys
import math
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import *



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


class TransformerModel(nn.Module):
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
                 vocab_size: int,
                 emb_size: int,
                 num_encoder_layers: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 device: torch.device = torch.device("cuda"),
                 pretrained_embedding=None,
                 freeze_embedding=False):
        super(TransformerModel, self).__init__()
        self.emb_size = emb_size
        self.device = device

        # Embedding + Positional Encoding
        self.tok_emb = TokenEmbedding(vocab_size, emb_size, pretrained_embedding, freeze_embedding)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Linear layer to convert the output of the transformer to the output vocabulary size
        self.generator = nn.Linear(emb_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.tok_emb.embedding.weight.requires_grad:
            self.tok_emb.embedding.weight.data.uniform_(-initrange, initrange)
        self.generator.bias.data.zero_()
        self.generator.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_pad_mask: torch.Tensor = None):
        src_emb = self.positional_encoding(self.tok_emb(src))

        # Create a mask to prevent the model from attending to the padding tokens
        src_seq_len = src.size(1)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src_seq_len).to(self.device)
        if src_pad_mask is None:
            src_pad_mask = (src == PAD_IDX).to(self.device)

        output = self.transformer_encoder(src_emb, mask=src_mask, src_key_padding_mask=src_pad_mask)

        return self.generator(output)
        
