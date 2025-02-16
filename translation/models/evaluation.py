import os
import sys
import torch

PROJECT_DIR = os.path.expanduser("~/Vietnamese-Poem-Generation/")
UTILS_DIR = os.path.join(PROJECT_DIR, "translation/utils")
sys.path.append(PROJECT_DIR)
sys.path.append(UTILS_DIR)
from constants import BOS_IDX, EOS_IDX, PAD_IDX, SRC_LANGUAGE, TGT_LANGUAGE, MAX_LEN
from data_loader import get_text_transforms

def translate_sentence(sentence, model, vocab, device, max_len=MAX_LEN):
    """
    Translate an input sentence using the seq2seq model with the correct transformation pipeline.
    
    Args:
        sentence (str): The sentence to translate (in Vietnamese).
        model (nn.Module): The Seq2Seq model.
        vocab (dict): The vocabulary for both languages.
        device (torch.device): The device to run the model on.
        max_len (int): The maximum length of the translated sentence.
        
    Returns:
        str: The predicted translated sentence.
    """
    model.eval()
    # Get the text transformation pipeline for the source language (Vietnamese)
    src_tensor = get_text_transforms(vocab)[SRC_LANGUAGE](sentence).unsqueeze(0).to(device)

    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)
    
    decoder_input = torch.tensor([[BOS_IDX]], device=device)
    translated_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            decoder_output, hidden = model.decoder(decoder_input, hidden)
        predicted_token = decoder_output.argmax(dim=-1)
        token_idx = predicted_token.item()
        if token_idx == EOS_IDX:
            break
        translated_tokens.append(token_idx)
        decoder_input = predicted_token

    translated_words = [vocab[TGT_LANGUAGE].get_itos()[token] for token in translated_tokens]
    return " ".join(translated_words)


def translate_tensor(src_tensor, model, vocab, device, max_len=MAX_LEN):
    """
    Translate a sentence represented as a tensor.

    Args:
        src_tensor (torch.Tensor): The input tensor with shape [1, seq_len].
        model (nn.Module): The Seq2Seq model.
        vocab (dict): The vocabulary of both languages.
        device (torch.device): The device to run the model on.
        max_len (int): The maximum length for the translated sentence.

    Returns:
        str: The translated sentence.
    """
    model.eval()
    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)

    decoder_input = torch.tensor([[BOS_IDX]], device=device)
    translated_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            decoder_output, hidden = model.decoder(decoder_input, hidden)
        predicted_token = decoder_output.argmax(dim=-1)
        token_idx = predicted_token.item()
        if token_idx == EOS_IDX:
            break
        translated_tokens.append(token_idx)
        decoder_input = predicted_token

    translated_words = [vocab[TGT_LANGUAGE].get_itos()[token] for token in translated_tokens]
    return " ".join(translated_words)


def tensor_to_sentence(token_ids, vocab, language):
    """
    Convert a list of token ids to a sentence, ignoring special tokens like BOS, EOS, PAD.

    Args:
        token_ids (list[int]): The list of token ids.
        vocab: The vocabulary of the language (with get_itos function).
        language (str): The language key (SRC_LANGUAGE or TGT_LANGUAGE).

    Returns:
        str: The corresponding sentence.
    """
    tokens = [vocab[language].get_itos()[token] for token in token_ids if token not in {BOS_IDX, EOS_IDX, PAD_IDX}]
    return " ".join(tokens)


def test_model(model, test_loader, vocab, device, num_examples=5):
    """
    Evaluate the model on the test_loader, print the source sentence, reference sentence, and predicted sentence.

    Args:
        model (nn.Module): The Seq2Seq model.
        test_loader (DataLoader): The dataloader for the test set.
        vocab (dict): The vocabulary of both languages.
        device (torch.device): The device to run the model on.
    """
    model.eval()
    example_count = 0
    with torch.no_grad():
        for src_batch, tgt_batch in test_loader:
            src_batch = src_batch.to(device)
            for i in range(src_batch.size(0)):
                src_tensor = src_batch[i:i+1]  # shape: [1, seq_len]
                input_sentence = tensor_to_sentence(src_tensor.squeeze(0).cpu().tolist(), vocab, SRC_LANGUAGE)
                tgt_tensor = tgt_batch[i].cpu().tolist()
                reference_sentence = tensor_to_sentence(tgt_tensor, vocab, TGT_LANGUAGE)
                predicted_sentence = translate_tensor(src_tensor, model, vocab, device)
                example_count += 1
                print(f"Example {example_count}:")
                print("Input:     ", input_sentence)
                print("Reference: ", reference_sentence)
                print("Prediction:", predicted_sentence)
                print("-" * 50)
                if example_count >= num_examples:
                    return
