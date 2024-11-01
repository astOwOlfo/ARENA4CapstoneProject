from transformers import PreTrainedTokenizerBase
import torch
import random
from dataclasses import dataclass
from beartype import beartype

@beartype
@dataclass(frozen=True)
class TokenizedSuffixesResult:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    indices: torch.Tensor

"""Copied from David Quarel's repo with his permission."""
@beartype
def safe_tokenize(suffixes : list[str] | str, 
                  tokenizer : PreTrainedTokenizerBase
) -> TokenizedSuffixesResult:
    tokenizer.pad_token = tokenizer.eos_token
    
    if isinstance(suffixes, str):    
        suffixes = [suffixes]
    
    if "Llama-2" in tokenizer.name_or_path:
        suffixes = ["🌍" + x for x in suffixes]
        space_token_id = tokenizer.convert_tokens_to_ids("▁")
        earth_token_id = tokenizer.convert_tokens_to_ids("🌍")
        
        suffix_tokens, attn_mask = tokenizer( suffixes,
                                              add_special_tokens=False,
                                              return_tensors="pt",
                                              padding=True ).values()
        
        assert torch.all(suffix_tokens[:, 0] == space_token_id), "llama2 has leading space token"
        assert torch.all(suffix_tokens[:, 1] == earth_token_id), "llama2 single token for 🌍"
        
        suffix_tokens = suffix_tokens[:, 2:]
        attn_mask = attn_mask[:, 2:]
        idx = attn_mask.sum(dim=-1) - 1 #-1, and another two more: one for the space token, one for the 🌍 token
    
    else: # models that do not add leading spaces
        x = tokenizer( suffixes,
                       add_special_tokens=False,
                       return_tensors="pt",
                       padding=True )
        suffix_tokens = x["input_ids"]
        attn_mask = x["attention_mask"]
        idx = attn_mask.sum(dim=-1) - 1

    assert torch.all(idx >= 0), "Attention mask has zeros, empty suffixes"
    
    return TokenizedSuffixesResult(
        input_ids=suffix_tokens,
        attention_mask=attn_mask,
        indices=idx
    )

@beartype
def safe_first_token_id_variants(text: str, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    if text.startswith(" "):
        variants = [text, text.lstrip(" ")]
    else:
        variants = [text, " " + text]

    token_ids = set()
    for variant in variants:
        try:
            token_id = safe_tokenize(variant, tokenizer=tokenizer).input_ids.flatten()[0].item()
        except AssertionError:
            print("WARNING: A placeholder token was inserted in one place because the tokenizer is weird. If this message isn't printed too often, this is not going to be statistically significant. If this message is printed too often, you will have to change something about tokenization for it not to be printed that often.")
            return [random.randint(0, tokenizer.vocab_size - 1)]
        
        token_ids.add(token_id)

    return list(token_ids)

"""
def safe_first_token_id(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    # I don't understand tokenization, I don't know why we need this line, but if we don't have it,
    # the safe_tokenize function that I didn't write throws a goofy error
    # hopefully, text is not "  " often enoguh for this to matter
    if text == "  ":
        return safe_first_token_id(text=" ", tokenizer=tokenizer)

    token = safe_tokenize(text, tokenizer=tokenizer).input_ids.flatten()[0].item()

    str_token = tokenizer.decode(token)
    if str_token == " " and text != " ":
        stripped_text = text.lstrip(" ")
        token = safe_tokenize(stripped_text, tokenizer=tokenizer).input_ids.flatten()[0].item()

    return token
"""

@beartype
def decode_if_bytes(text: str | bytes) -> str:
    if isinstance(text, str):
        return text
    
    try:
        return text.decode()
    except UnicodeDecodeError:
        print(f"WARNING: Cannot decode bytes {text} with utf-8.")
        return ""