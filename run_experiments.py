from dataset_synthesis import make_translated_token_dataset, load_multiligual_dataset, \
                              TokenTranslations, Language
from logit_lens import logit_lens_probs_of_translated_tokens, logit_lens_top_k
from plotting import plot_logit_lens_probs_of_translated_tokens, plot_logit_lens_top_k, \
                     plot_translated_token_dataset

from anthropic import Anthropic
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import huggingface_hub
import torch
import random
import gc
from beartype import beartype

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using", device)

@beartype
def run_translated_token_logit_lens_experiments(
        languages: list[Language],
        model_names: list[str],
        n_prompts: int,
        multilingual_prompts_dataset_name: str,
        batch_size: int = 16,
        k_for_logit_lens_top_k: int = 16,
        minimal_prompt_length_characters: int = 64,
        anthropic_api_key: str = "", # you can pass "" if no api calls will be made because all
                                     # api call results have already been cached
        huggingface_token: str | None = None,
        dataset_shuffling_seed: int = 42
    ):
    if huggingface_token is not None:
        huggingface_hub.login(huggingface_token)

    anthropic_client = Anthropic(api_key=anthropic_api_key)

    for model_name in model_names:
        if "model" in locals():
            del model
            gc.collect()
            torch.cuda.empty_cache()

        model = HookedTransformer.from_pretrained(model_name, dtype=torch.bfloat16, device=device)

        logit_lens_checkpoints = [ get_act_name("resid_post", layer)
                                   for layer in range(model.cfg.n_layers) ]

        for prompt_language in languages:
            plot_directory = f"plots/{multilingual_prompts_dataset_name}/" \
                             f"{model_name.replace('/', '-')}" \
                             f"/from-{prompt_language}-to-{'-'.join(languages)}/"

            dataset: list[str] = load_multiligual_dataset(
                dataset_name = multilingual_prompts_dataset_name,
                language = prompt_language
            )
            random.seed(dataset_shuffling_seed)
            random.shuffle(dataset)

            translated_token_dataset: list[TokenTranslations] = make_translated_token_dataset(
                texts = dataset,
                size = n_prompts,
                tokenizer = model.tokenizer,
                source_language = prompt_language,
                target_languages = languages,
                anthropic_client = anthropic_client,
                minimal_text_length = minimal_prompt_length_characters
            )

            plot_translated_token_dataset(
                translated_token_dataset,
                tokenizer = model.tokenizer,
                languages = languages,
                save_filename = f"{plot_directory}/translated_token_dataset.html"
            )

            top_k = logit_lens_top_k(
                model = model,
                prompts = [datapoint.text for datapoint in translated_token_dataset],
                checkpoints = logit_lens_checkpoints,
                k = k_for_logit_lens_top_k,
                batch_size = batch_size
            )

            plot_logit_lens_top_k(
                top_k = top_k,
                checkpoints = logit_lens_checkpoints,
                tokenizer = model.tokenizer,
                translated_token_dataset = translated_token_dataset,
                save_filename = f"{plot_directory}/logit_lens_top_k.html"
            )

            probs = logit_lens_probs_of_translated_tokens(
                model = model,
                translated_token_dataset = translated_token_dataset,
                target_languages = languages,
                checkpoints = logit_lens_checkpoints,
                batch_size = batch_size
            )

            plot_logit_lens_probs_of_translated_tokens(
                model_name = model_name,
                probs = probs,
                source_language = prompt_language,
                target_languages = languages,
                checkpoints = logit_lens_checkpoints,
                save_filename = f"{plot_directory}/logit_lens_probabilities_of_translated_tokens.html"
            )

if __name__ == "__main__":
    run_translated_token_logit_lens_experiments(
        languages = ["English", "French", "Russian"],
        model_names = ["mistral-7b", "google/gemma-2-2b", "meta-llama/llama-2-7b-hf"],
        n_prompts = 16,
        multilingual_prompts_dataset_name = "opus_books",
        batch_size = 1, # reduce batch size if you get out of memory (both gpu and cpu memory)
        # anthropic_api_key = "INSERT ANTHROPIC KEY",
        huggingface_token = "INSERT HUGGINGFACE TOKEN",
    )

    """
    run_translated_token_logit_lens_experiments(
        languages = ["English", "Chinese", "Kazakh", "Russian"],
        model_names = ["Qwen/Qwen-7b", "Qwen/Qwen2-7b", "mistral-7b"],
        n_prompts = 16,
        multilingual_prompts_dataset_name = "wmt",
        batch_size = 1, # reduce batch size if you get out of memory (both gpu and cpu memory)
        anthropic_api_key = "INSERT ANTHROPIC API KEY",
        huggingface_token = "INSERT HUGGINGFACE TOKEN",
    )
    """
