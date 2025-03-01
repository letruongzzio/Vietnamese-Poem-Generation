import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *

PROJECT_ROOT = os.path.expanduser('~/vietnamese-poem-generation')
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from constants import *
sys.path.append(UTILS_DIR)
from tokenization import *


def sample_with_temperature(logits, temperature=1.0):
    """
    Sample from a categorical distribution with temperature.

    Args:
        logits (Tensor): Logits from the model.
        temperature (float): Temperature value (default: 1.0).

    Returns:
        int: Sampled token index.
    """
    if temperature != 1.0:
        logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)
    sampled_index = torch.multinomial(probabilities, 1).item()
    return sampled_index

def decode_token_ids(token_ids, vocab):
    """
    Convert a list of token IDs to a list of tokens using the vocabulary.

    Args:
        token_ids (list): List of token IDs.
        vocab (Vocab): Vocabulary object.

    Returns:
        list: List of tokens.
    """
    return [vocab.get_itos()[token_id] for token_id in token_ids]

def inference(model, vocab, input_text, device, tokenizer=custom_tokenize, max_seq_len=50, temperature=1.0):
    """
    Generate text using the provided model and input text.

    Args:
        model (nn.Module): The trained model.
        vocab (Vocab): Vocabulary object.
        input_text (str): Input text.
        device (torch.device): Device to run the model on.
        tokenizer (callable): Tokenizer function (default: custom_tokenize).
        max_seq_len (int): Maximum sequence length (default: 50).
        temperature (float): Temperature value for sampling (default: 1.0).

    Returns:
        str: Generated text.
    """
    model.eval()
    model = model.to(device)
    
    # Convert to lowercase
    input_text = input_text.lower().strip()

    # Add <sos> if not present
    if not input_text.startswith('<sos>'):
        input_text = '<sos> ' + input_text

    with torch.no_grad():
        input_tokens = tokenizer(input_text)
        input_ids = [vocab[token] for token in input_tokens]
        generated_ids = input_ids.copy()

        for _ in range(max_seq_len):
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            logits = model(input_tensor)[0, -1, :]
            next_token_id = sample_with_temperature(logits, temperature)
            generated_ids.append(next_token_id)
            input_ids = generated_ids.copy()
            if next_token_id == EOS_IDX:
                break

        # Convert token IDs back to string
        generated_tokens = decode_token_ids(generated_ids, vocab)

        # (Optional) Remove <unk> from the result if you don't want to display them
        generated_tokens = [t for t in generated_tokens if t != '<unk>']

        # Join tokens into a string
        text = ' '.join(generated_tokens)

        # Remove <sos>, <eos>, then split lines by <eol>
        text = text.replace('<sos>', '').replace('<eos>', '')
        lines = text.split('<eol>')

        # Remove extra whitespace
        lines = [line.strip() for line in lines if line.strip()]

        # Helper function to capitalize the first letter (ignore non-letter characters)
        def capitalize_first_alpha(s):
            chars = list(s)
            for i in range(len(chars)):
                if chars[i].isalpha():
                    chars[i] = chars[i].upper()
                    break
            return ''.join(chars)

        # Capitalize the first letter of each line
        capitalized_lines = [capitalize_first_alpha(line) for line in lines]

        # Join lines with newline character
        final_text = '\n'.join(capitalized_lines)

        return final_text
