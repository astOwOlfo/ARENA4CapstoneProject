from dataset_synthesis import TokenTranslations, Language

from transformer_lens import HookedTransformer
from torch import Tensor, cat, inference_mode
from torch.nn.functional import pad
from itertools import pairwise, chain
from more_itertools import chunked
from tqdm import tqdm
from dataclasses import dataclass
from typing import Iterable
from beartype import beartype
from jaxtyping import jaxtyped, Float, Int

Checkpoint = str

@beartype
def pairwise_distinct(xs: Iterable) -> bool:
    xs = list(xs)
    hashable_xs = [ (tuple(x) if isinstance(x, list) else x)
                    for x in xs ]
    return len(set(hashable_xs)) == len(hashable_xs)

@beartype
def all_equal(xs: Iterable) -> bool:
    return all(x == y for x, y in pairwise(xs))

@jaxtyped(typechecker=beartype)
def concatenate_and_pad(tensors: list[Tensor], dim: int, pad_with: float = 0.) -> Tensor:
    assert len(tensors) > 0
    assert all_equal(tensor.ndim for tensor in tensors)

    ndim = tensors[0].ndim
    max_size = [ max(tensor.size(i) for tensor in tensors)
                 for i in range(ndim) ]
    padded_tensors = []
    for tensor in tensors:
        pad_size = []
        for i in range(ndim):
            if i == dim:
                pad_size += [0, 0]
                continue
            pad_size += [0, max_size[i] - tensor.size(i)]
        padded_tensors.append(pad(tensor, pad_size, value=pad_with))

    return cat(padded_tensors, dim=dim)

@jaxtyped(typechecker=beartype)
@inference_mode()
def logit_lens_probs(
        model: HookedTransformer,
        prompts: list[str],
        checkpoints: list[Checkpoint]
    ) -> dict[Checkpoint, Float[Tensor, "batch sequence_length vocabulary_size"]]:

    _, cache = model.run_with_cache(prompts)

    probs = {}
    for checkpoint in checkpoints:
        activations = cache[checkpoint]
        logits = model.unembed(model.ln_final(activations))
        logits = logits.to("cpu")
        probs[checkpoint] = logits.softmax(-1) # log softmax along the vocabulary_size dim

    return probs

@beartype
def logit_lens_probs_of_translated_tokens(
        model: HookedTransformer,
        translated_token_dataset: list[TokenTranslations],
        target_languages: list[Language],
        checkpoints: list[Checkpoint],
        batch_size: int = 8
    ) -> dict[tuple[Language, Checkpoint], list[float]]:

    if len(translated_token_dataset) > batch_size:
        probs: list[dict[tuple[Language, Checkpoint], list[float]]] = [
            logit_lens_probs_of_translated_tokens( model = model,
                                                   translated_token_dataset = batch,
                                                   target_languages = target_languages,
                                                   checkpoints = checkpoints,
                                                   batch_size = batch_size )
            for batch in tqdm( list(chunked(translated_token_dataset, batch_size)),
                               desc="computing logit lens" )
        ]

        return { key: list(chain.from_iterable(p[key] for p in probs))
                 for key in probs[0].keys() }

    all_probs: dict[Checkpoint, Float[Tensor, "batch sequence_length vocabulary_size"]] = \
        logit_lens_probs( model = model,
                          prompts = [datapoint.text for datapoint in translated_token_dataset],
                          checkpoints = checkpoints )

    batch_size, sequence_length, vocabulary_size = all_probs[checkpoints[0]].shape

    wanted_probs: dict[tuple[Language, Checkpoint], list[float]] = {
        (language, checkpoint): []
        for language in target_languages
        for checkpoint in checkpoints   
    }
    for checkpoint in checkpoints:
        for i_batch, datapoint in enumerate(translated_token_dataset):
            for position, token_translations in enumerate(datapoint.token_translations):
                if token_translations is None:
                    continue

                assert set(token_translations.keys()) == set(target_languages)

                if not pairwise_distinct(token_translations.values()):
                    continue

                for language in target_languages:
                    token_variants: list[int] = token_translations[language]
                    prob = all_probs[checkpoint][i_batch, position, token_variants].mean().item()
                    wanted_probs[language, checkpoint].append(prob)

    return wanted_probs

@dataclass
class LogitLensTopK:
    token_ids: Int  [Tensor, "batch_size sequence_length k"]
    probs:     Float[Tensor, "batch_size sequence_length k"]

def logit_lens_top_k(
        model: HookedTransformer,
        prompts: list[str],
        checkpoints: list[Checkpoint],
        k: int,
        batch_size: int = 16
    ) -> dict[Checkpoint, LogitLensTopK]:

    if len(prompts) > batch_size:
        result_batches = [ logit_lens_top_k( model = model,
                                             prompts = batch,
                                             checkpoints = checkpoints,
                                             k = k,
                                             batch_size = batch_size )
                           for batch in tqdm( list(chunked(prompts, batch_size)),
                                              desc = "computing logit lens" ) ]
                                            
        return {
            checkpoint: LogitLensTopK(
                token_ids = concatenate_and_pad(
                    [batch[checkpoint].token_ids for batch in result_batches],
                    dim = 0 # 0 is batch dim
                ),
                probs = concatenate_and_pad(
                    [batch[checkpoint].probs     for batch in result_batches],
                    dim = 0 # 0 is batch dim
                )
            )
            for checkpoint in checkpoints
        }

    all_probs: Float[Tensor, "batch sequence_length vocabulary_size"] = \
        logit_lens_probs(model=model, prompts=prompts, checkpoints=checkpoints)

    result = {}
    for checkpoint, probs in all_probs.items():
        top_k_probs, top_k_token_ids = probs.topk(k=k, dim=-1) # topk along vocabulary_size dim
        result[checkpoint] = LogitLensTopK(token_ids=top_k_token_ids, probs=top_k_probs)

    return result
