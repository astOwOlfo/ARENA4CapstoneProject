# %%

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase
import huggingface_hub
import csv
import random
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import numpy as np
import gc
from dataclasses import dataclass
from jaxtyping import jaxtyped, Float
from typing import Literal
from beartype import beartype

# %%

huggingface_hub.login(token="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")

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)

# %%

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
def start_token_id_variants(string: str, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    # TO DO: also handle capitalization

    result = []
    
    start_token_id = safe_tokenize(string, tokenizer).input_ids[0, 0].item()
    assert string.startswith(tokenizer.decode(start_token_id)), \
           "Some cursed tokenization stuff happened, it is not supposed to happen."
    result.append(start_token_id)
    
    start_token_id_variant = safe_tokenize(" " + string, tokenizer).input_ids[0, 0].item()
    # if string.startswith(tokenizer.decode(start_token_id_variant)):
    result.append(start_token_id_variant)

    result = list(set(result))

    return result

LANGUAGES = ["en", "de", "fr", "ru", "zh"]
LANGUAGE_NAMES = {"en": "English", "de": "Deusch", "fr": "Français", "ru": "Русский", "zh": "中文"}
Language = Literal["en", "de", "fr", "ru", "zh"]

@beartype
class LanguageDataset:
    data: dict[Language, dict[str, list[dict]]]

    def __init__(self, path="/root/llm-latent-language/data/") -> None:
        self.data = {}
        for language in LANGUAGES:
            with open(f"{path}/langs/{language}/clean.csv", "r") as f:
                data = list(csv.DictReader(f))
                data = {x["word_original"]: x for x in data}
                self.data[language] = data

    def translatable_english_words(self, translatable_to: Language | list[Language]) -> list[str]:
        if isinstance(translatable_to, str):
            translatable_to = [translatable_to]

        return list(set.intersection(*(
            set(self.data[language].keys())
            for language in translatable_to
        )))

    def translate_from_english(self, english_word: str, target_language: Language) -> str:
        if english_word not in self.data[target_language]:
            raise KeyError(f"Translation of english word '{english_word}' to language '{target_language}' not found.")
        return self.data[target_language][english_word]["word_translation"]
    
    def get_cloze_and_word(self, english_word: str, language: Language) -> tuple[str, str]:
        if english_word not in self.data[language]:
            raise KeyError(f"Translation of english word '{english_word}' to language '{language}' not found.")
        cloze = self.data[language][english_word]["blank_prompt_translation_masked"]
        word = self.data[language][english_word]["word_translation"]
        stripped_cloze = cloze.strip()
        suffixes = [".", "。", '"', "'"]
        while any(stripped_cloze.endswith(suffix) for suffix in suffixes):
            for suffix in suffixes:
                stripped_cloze = stripped_cloze.removesuffix(suffix)
        # TO DO: count number of invalid examples
        # if not stripped_cloze.lower().endswith(word.lower()):
        #     raise ValueError(f"Cloze '{cloze}' must end with '{word}' after stripping but doesn't")
        stripped_cloze = stripped_cloze.removesuffix(word)
        return stripped_cloze, word

@beartype
@dataclass
class LogitLensDatasetEntry: # how to name this class?
    prompt: str
    logit_lens_token_ids: dict[str, list[int]]

@beartype
def make_repetition_dataset(
        language_dataset: LanguageDataset,
        tokenizer: PreTrainedTokenizerBase,
        size: int,
        prompt_language: Language,
        logit_lens_languages: list[Language],
        n_few_shot_examples: int = 4
    ) -> list[LogitLensDatasetEntry]:

    english_words = \
        language_dataset.translatable_english_words([prompt_language] + logit_lens_languages)
    
    prompt_language_name = LANGUAGE_NAMES[prompt_language]

    dataset: list[LogitLensDatasetEntry] = []

    for _ in range(size):
        prompt = ""
        for _ in range(n_few_shot_examples):
            english_word = random.choice(english_words)
            word = language_dataset.translate_from_english(english_word, prompt_language)
            prompt += f'{prompt_language_name}: "{word}" - {prompt_language_name}: "{word}"\n'
        english_word = random.choice(english_words)
        word = language_dataset.translate_from_english(english_word, prompt_language)
        prompt += f'{prompt_language_name}: "{word}" - {prompt_language_name}: "'
        
        logit_lens_token_ids = {}
        for language in logit_lens_languages:
            word = language_dataset.translate_from_english(english_word, language)
            logit_lens_token_ids[language] = start_token_id_variants(word, tokenizer=tokenizer)
            
        dataset.append(LogitLensDatasetEntry( prompt = prompt,
                                              logit_lens_token_ids = logit_lens_token_ids ))

    return dataset

@beartype
def make_translation_dataset(
        language_dataset: LanguageDataset,
        tokenizer: PreTrainedTokenizerBase,
        size: int,
        prompt_source_language: Language,
        prompt_target_language: Language,
        logit_lens_languages: list[Language],
        n_few_shot_examples: int = 4
    ) -> list[LogitLensDatasetEntry]:

    english_words = language_dataset.translatable_english_words(
        [prompt_source_language] + [prompt_target_language] + logit_lens_languages
    )
    
    source_language_name = LANGUAGE_NAMES[prompt_source_language]
    target_language_name = LANGUAGE_NAMES[prompt_target_language]

    dataset: list[LogitLensDatasetEntry] = []

    for _ in range(size):
        prompt = ""
        for _ in range(n_few_shot_examples):
            english_word = random.choice(english_words)
            word_in_source_language = \
                language_dataset.translate_from_english(english_word, prompt_source_language)
            word_in_target_language = \
                language_dataset.translate_from_english(english_word, prompt_target_language)
            prompt += f'{source_language_name}: "{word_in_source_language}"' \
                      f' - {target_language_name}: "{word_in_target_language}"\n'
        english_word = random.choice(english_words)
        word_in_source_language = \
            language_dataset.translate_from_english(english_word, prompt_source_language)
        word_in_target_language = \
            language_dataset.translate_from_english(english_word, prompt_target_language)
        prompt += f'{source_language_name}: "{word_in_source_language}"' \
                  f' - {target_language_name}: "'
        
        logit_lens_token_ids = {}
        for language in logit_lens_languages:
            word = language_dataset.translate_from_english(english_word, language)
            logit_lens_token_ids[language] = start_token_id_variants(word, tokenizer=tokenizer)
            
        dataset.append(LogitLensDatasetEntry( prompt = prompt,
                                              logit_lens_token_ids = logit_lens_token_ids ))

    return dataset

@beartype
def make_cloze_dataset(
        language_dataset: LanguageDataset,
        tokenizer: PreTrainedTokenizerBase,
        size: int,
        prompt_language: Language,
        logit_lens_languages: list[Language],
        n_few_shot_examples: int = 2
    ) -> list[LogitLensDatasetEntry]:

    english_words = \
        language_dataset.translatable_english_words([prompt_language] + logit_lens_languages)
    
    dataset: list[LogitLensDatasetEntry] = []

    for _ in range(size):
        prompt = ""
        for _ in range(n_few_shot_examples):
            english_word = random.choice(english_words)
            cloze, word = language_dataset.get_cloze_and_word(english_word, prompt_language)
            prompt += f'{cloze}{word}"\n'    
        english_word = random.choice(english_words)
        cloze, word = language_dataset.get_cloze_and_word(english_word, prompt_language)
        prompt += cloze

        logit_lens_token_ids = {}
        for language in logit_lens_languages:
            word = language_dataset.translate_from_english(english_word, language)
            logit_lens_token_ids[language] = start_token_id_variants(word, tokenizer=tokenizer)
            
        dataset.append(LogitLensDatasetEntry( prompt = prompt,
                                              logit_lens_token_ids = logit_lens_token_ids ))

    return dataset

@beartype
def make_synthetic_dataset(
        *args,
        dataset_type: Literal["repetition", "translation", "cloze"],
        **kwargs
    ) -> list[LogitLensDatasetEntry]:

    make_dataset_function = {
        "repetition": make_repetition_dataset,
        "translation": make_translation_dataset,
        "cloze": make_cloze_dataset
    }[dataset_type]
    
    return make_dataset_function(*args, **kwargs)

# %%

"""
# MANUALLY CHECK THAT THE DATASET WAS GENERATED PROPERLY
language_dataset = LanguageDataset()
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen-7B",
    dtype=torch.bfloat16,
    device=device
)

for language in LANGUAGES:
    print(f"{language=}")
    dataset = make_cloze_dataset(language_dataset=language_dataset, tokenizer=model.tokenizer, size=100000, prompt_language=language, logit_lens_languages=["ru"])
    for entry in dataset:
        print("=========== ENTRY ===========")
        print("PROMPT:", entry.prompt)
        for language, token_ids in entry.logit_lens_token_ids.items():
            print(f"LOGIT LENS TOKENS IN LANGUAGE {language}:", model.tokenizer.batch_decode(token_ids))
"""

# %%

@jaxtyped(typechecker=beartype)
def plot_tensor_dict(
        data: dict[str, Float[torch.Tensor, "batch layer"]],
        title: str = "Logit lens in multiple languages.",
        x_axis_title: str = "layer",
        y_axis_title: str = "logit lens probability",
        error_bar_confidence: float = 0.95,
        y_axis_range: tuple = (0, 1)
    ):

    fig = make_subplots()

    for key, tensor in data.items():
        # Ensure tensor is on CPU and convert to numpy
        tensor_np = tensor.cpu().numpy()
        
        # Calculate mean and standard error
        mean = np.mean(tensor_np, axis=0)
        std_error = np.std(tensor_np, axis=0) / np.sqrt(tensor_np.shape[0])
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + error_bar_confidence) / 2)
        margin_of_error = z_score * std_error
        
        # Create x-axis values (layer numbers)
        x = list(range(1, len(mean) + 1))
        
        # Add trace for line and error bars
        fig.add_trace(go.Scatter(
            x=x,
            y=mean,
            error_y=dict(
                type='data',
                array=margin_of_error,
                visible=True
            ),
            mode='lines+markers',
            name=key
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        yaxis_range=y_axis_range,
        legend_title="Models",
        hovermode="x unified"
    )

    fig.show()

# %%

@jaxtyped(typechecker=beartype)
def logit_lens(
        model: HookedTransformer,
        dataset: list[LogitLensDatasetEntry],
        checkpoints: list[str | tuple[str, int]] = None
    ) -> dict[str, Float[torch.Tensor, "batch layer"]]:
    
    if checkpoints is None:
        checkpoints = [("resid_post", layer) for layer in range(model.cfg.n_layers)]

    assert len(dataset) > 0

    keys = list(dataset[0].logit_lens_token_ids.keys())
    assert all(set(entry.logit_lens_token_ids.keys()) == set(keys) for entry in dataset), \
           "logit_lens_token_ids of all entries of the dataset should have the samke keys"
    all_logit_lens_probs = {
        key: torch.full(size=(len(dataset), model.cfg.n_layers), fill_value=42.42)
        for key in keys
    }
    
    for i_entry, entry in enumerate(tqdm(dataset, desc="running logit lens")):
        _, cache = model.run_with_cache(entry.prompt)
        # print(entry.prompt)
        # for key, token_ids in entry.logit_lens_token_ids.items():
        #     print(key, token_ids, model.tokenizer.batch_decode(token_ids))
        for i_checkpoint, checkpoint in enumerate(checkpoints):
            activations = cache[checkpoint]

            # squeeze batch, take activations only at the last token
            activations = activations.squeeze(0)[-1, :]

            normalized_activations = model.ln_final(activations)
            logit_lens_logits = model.unembed(normalized_activations)
            logit_lens_probs = logit_lens_logits.softmax(dim=0)

            # print(layer, model.tokenizer.batch_decode(logit_lens_probs.topk(10).indices.tolist()))

            for key, token_ids in entry.logit_lens_token_ids.items():
                # print(key, token_ids, model.tokenizer.batch_decode(token_ids))
                prob_for_token_ids = logit_lens_probs[token_ids].sum().item()
                all_logit_lens_probs[key][i_entry, i_checkpoint] = prob_for_token_ids
    
    return all_logit_lens_probs

# %%

def compute_and_plot_logit_lens_probabilities_in_different_languages(
        sample_size: int,
        model_names: list[str],
        dataset_types: list[Literal["translation", "repetition", "cloze"]],
        languages: list[Language],
        device: str,
    ) -> None:

    language_dataset = LanguageDataset()

    for model_name in model_names:
        model = HookedTransformer.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device=device
        )

        for dataset_type in dataset_types:
            if dataset_type != "translation":
                possible_dataset_language_arguments = [
                    { "prompt_language": language }
                    for language in languages
                ]
            else:
                possible_dataset_language_arguments = [
                    { "prompt_source_language": source_language,
                    "prompt_target_language": target_language }
                    for source_language in languages
                    for target_language in languages
                    if source_language != target_language
                ]

            for dataset_language_arguments in possible_dataset_language_arguments:
                dataset = make_synthetic_dataset(
                    dataset_type=dataset_type,
                    language_dataset=language_dataset,
                    tokenizer=model.tokenizer,
                    size=sample_size,
                    logit_lens_languages=LANGUAGES,
                    **dataset_language_arguments
                )

                result = logit_lens(model=model, dataset=dataset)
                
                plot_tensor_dict(
                    result,
                    title=f"{model_name} logit lens probabilities in different languages for {dataset_type} dataset with {dataset_language_arguments}."
                )

        # make sure we don't get out of memory errors
        del model
        torch.cuda.empty_cache()
        gc.collect()

# %%

compute_and_plot_logit_lens_probabilities_in_different_languages(
    sample_size = 100,
    # model_names = ["Qwen/Qwen-7B", "meta-llama/Llama-2-7b-hf", "mistral-7b", "google/gemma-2-7b"],
    model_names = ["google/gemma-2-9b"],
    dataset_types = ["translation", "repetition", "cloze"],
    languages = LANGUAGES,
    device = device
)

# %%
