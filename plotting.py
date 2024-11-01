from dataset_synthesis import TokenTranslations, Language, Checkpoint
from logit_lens import LogitLensTopK
from tokenization import decode_if_bytes

from transformers import PreTrainedTokenizerBase
from scipy import stats
from pathlib2 import Path
from copy import deepcopy
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from beartype import beartype

@beartype
def create_parent_directories(filename: str) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

@beartype
def save_or_show(fig: go.Figure, save_filename: str | None = None) -> None:
    if save_filename is None:
        fig.show()
    else:
        assert save_filename.endswith(".html")
        create_parent_directories(save_filename)
        fig.write_html(save_filename)
        print(f"Saved figure to '{save_filename}'.")

@beartype
def plot_logit_lens_probs_of_translated_tokens(
        model_name: str,
        probs: dict[tuple[Language, Checkpoint], list[float]],
        source_language: Language,
        target_languages: list[Language],
        checkpoints: list[Checkpoint],
        error_bar_confidence: float = 0.95,
        save_filename: str | None = None
    ) -> None:

    assert set(probs.keys()) == set((language, checkpoint)
                                    for language in target_languages
                                    for checkpoint in checkpoints)

    fig = go.Figure()
    fig.update_layout(title=f"{model_name} logit lens probs of translated tokens. "
                            f"Original prompt in {source_language}",
                      xaxis=dict(title="layer"),
                      yaxis=dict(title="logit lens prob", range=[0, 1]))

    for language in target_languages:
        mean_probs = [np.mean(probs[language, checkpoint]) for checkpoint in checkpoints]
        
        # Calculate confidence intervals
        confidence_intervals = [stats.t.interval( error_bar_confidence,
                                                  len(probs[language, checkpoint]) - 1,
                                                  loc = np.mean(probs[language, checkpoint]),
                                                  scale = stats.sem(probs[language, checkpoint]) )
                                for checkpoint in checkpoints]
        
        # Extract lower and upper bounds
        lower_bounds = [ci[0] for ci in confidence_intervals]
        upper_bounds = [ci[1] for ci in confidence_intervals]
        
        # Calculate error bar values
        error_y = dict(
            type='data',
            symmetric=False,
            array=[ub - m for ub, m in zip(upper_bounds, mean_probs)],
            arrayminus=[m - lb for lb, m in zip(lower_bounds, mean_probs)],
            visible=True
        )

        fig.add_scatter(x=list(range(len(checkpoints))), y=mean_probs, 
                        name=f"{language} tokens", 
                        error_y=error_y)

    save_or_show(fig, save_filename)

@beartype
def plot_logit_lens_top_k(
        top_k: dict[str, LogitLensTopK],
        checkpoints: list[str],
        tokenizer: PreTrainedTokenizerBase,
        translated_token_dataset: list[TokenTranslations],
        save_filename: str | None = None
    ) -> None:

    # Extract batch_size, sequence_length, k from any of the LogitLensTopK objects
    sample_logit_lens = next(iter(top_k.values()))
    batch_size, sequence_length, k = sample_logit_lens.token_ids.shape

    # Determine the maximum number of valid positions across all batches
    max_valid_positions = 0
    for b in range(batch_size):
        token_translations = translated_token_dataset[b].token_translations
        valid_positions = [
            s for s in range(sequence_length)
            if s < len(token_translations) and token_translations[s] is not None
        ]
        max_valid_positions = max(max_valid_positions, len(valid_positions))

    # Process the first batch to get initial subplot titles
    b = 0  # First batch
    token_translations_data = translated_token_dataset[b]
    token_translations = token_translations_data.token_translations
    text_prompt = token_translations_data.text

    # Valid positions for this batch element
    valid_positions = [
        s for s in range(sequence_length)
        if s < len(token_translations) and token_translations[s] is not None
    ]

    batch_subplot_titles = []
    # Iterate over each valid sequence position
    for idx, s in enumerate(valid_positions):
        translations = token_translations[s]  # dict[Language, int]
        # Build the title from translations
        translation_text = ', '.join(
            f"{lang}: {tokenizer.convert_ids_to_tokens(tid)[0]}"
            for lang, tid in translations.items()
        )
        batch_subplot_titles.append(translation_text)

    # Pad the titles to match max_valid_positions
    initial_subplot_titles = batch_subplot_titles + [''] * (max_valid_positions - len(batch_subplot_titles))

    # Compute vertical_spacing
    if max_valid_positions > 1:
        max_vertical_spacing = 1.0 / (max_valid_positions - 1)
        vertical_spacing = max_vertical_spacing * 0.9  # Slightly less than the maximum allowed
    else:
        vertical_spacing = 0  # Only one row, no spacing needed

    # Create a figure with subplots for the maximum number of valid positions
    fig = make_subplots(
        rows=max_valid_positions,
        cols=1,
        shared_xaxes=False,
        # vertical_spacing=vertical_spacing,  # Adjusted vertical spacing
        subplot_titles=initial_subplot_titles
    )

    # Capture the initial subplot title annotations
    subplot_title_annotations = fig.layout.annotations[:max_valid_positions]

    # Keep track of trace indices and annotations for each batch element
    batch_traces_indices_list = []
    batch_annotations_list = []

    # Now proceed to process each batch
    for b in range(batch_size):
        batch_trace_indices = []
        batch_heatmap_annotations = []
        batch_subplot_titles = []
        token_translations_data = translated_token_dataset[b]
        token_translations = token_translations_data.token_translations
        text_prompt = token_translations_data.text

        # Valid positions for this batch element
        valid_positions = [
            s for s in range(sequence_length)
            if s < len(token_translations) and token_translations[s] is not None
        ]

        # Iterate over each valid sequence position
        for idx, s in enumerate(valid_positions):
            row = idx + 1  # Subplot row index (1-based)
            translations = token_translations[s]  # dict[Language, int]
            # Build the title from translations
            translation_text = ', '.join(
                f"{lang}: {tokenizer.convert_ids_to_tokens(tid)[0]}"
                for lang, tid in translations.items()
            )
            batch_subplot_titles.append(translation_text)

            z = []
            annotations = []
            # Iterate over each checkpoint
            for cp_idx, cp in enumerate(checkpoints):
                logit_lens = top_k[cp]
                token_ids = logit_lens.token_ids[b, s, :].tolist()
                probs = logit_lens.probs[b, s, :].tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                z.append(probs)
                for k_idx, (token_id, token, prob) in enumerate(zip(token_ids, tokens, probs)):
                    annotation_text = f"{token}<br>{prob:.2f}"
                    # Check if the token matches any translation
                    assert isinstance(token_id, int)
                    assert all(isinstance(tid, list) for tid in translations.values())
                    match_languages = [
                        lang for lang, tid in translations.items() if token_id in tid
                    ]
                    if match_languages:
                        match_text = ', '.join(match_languages)
                        annotation_text += f"<br>MATCHES {match_text} TRANSLATION"
                    # Adjust x and y positions to match the heatmap cells
                    annotations.append(dict(
                        x=k_idx,
                        y=cp_idx,
                        xref=f'x{row}',
                        yref=f'y{row}',
                        text=annotation_text,
                        showarrow=False,
                        font=dict(color='white' if prob < 0.5 else 'black'),
                        align='center'
                    ))
            x = [f"Top {i+1}" for i in range(k)]
            y = checkpoints
            heatmap = go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='Viridis',
                colorbar=dict(title="Probability") if idx == 0 else dict(),
                showscale=idx == 0,
                visible=b == 0  # Only show traces for the first batch element initially
            )
            fig.add_trace(heatmap, row=row, col=1)
            batch_heatmap_annotations.extend(annotations)
            trace_index = len(fig.data) - 1
            batch_trace_indices.append(trace_index)

        # Handle positions without valid data (add empty traces)
        for idx in range(len(valid_positions), max_valid_positions):
            row = idx + 1
            # Add an invisible heatmap to maintain the subplot structure
            heatmap = go.Heatmap(
                z=[[0]*k]*len(checkpoints),
                x=[f"Top {i+1}" for i in range(k)],
                y=checkpoints,
                colorscale='Viridis',
                showscale=False,
                visible=False  # Always hidden
            )
            fig.add_trace(heatmap, row=row, col=1)
            trace_index = len(fig.data) - 1
            batch_trace_indices.append(trace_index)
            batch_subplot_titles.append('')  # Empty title for unused subplots

        # Create batch_subplot_annotations
        batch_subplot_annotations = []
        for idx, anno in enumerate(subplot_title_annotations):
            # Create a copy of the annotation
            new_anno = deepcopy(anno)
            if idx < len(batch_subplot_titles):
                new_anno['text'] = batch_subplot_titles[idx]
            else:
                new_anno['text'] = ''
            batch_subplot_annotations.append(new_anno)

        # Combine annotations
        batch_annotations = batch_subplot_annotations + batch_heatmap_annotations

        # Store annotations
        batch_annotations_list.append(batch_annotations)

        # Store trace indices for this batch
        batch_traces_indices_list.append(batch_trace_indices)

    total_traces = len(fig.data)

    # Set figure height
    subplot_height = 100 * len(checkpoints)  # Adjust as needed
    total_height = max_valid_positions * subplot_height
    fig.update_layout(
        annotations=batch_annotations_list[0],
        height=total_height,
        title_text=translated_token_dataset[0].text
    )

    # Create dropdown buttons for batch selection
    buttons = []
    for b in range(batch_size):
        visibility = [False] * total_traces
        # Update visibility for valid traces
        for idx in batch_traces_indices_list[b]:
            visibility[idx] = True
        # Update annotations and titles
        button = dict(
            label=f"Prompt {b}",
            method="update",
            args=[
                {"visible": visibility},
                {
                    "annotations": batch_annotations_list[b],
                    "title.text": translated_token_dataset[b].text + "<br>"
                                    + str(tokenizer.tokenize(translated_token_dataset[b].text)),
                }
            ]
        )
        buttons.append(button)

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.5,
            xanchor='center',
            y=1.15,
            yanchor='top'
        )]
    )

    save_or_show(fig, save_filename)

@beartype
def plot_translated_token_dataset(
    translated_token_dataset: list[TokenTranslations],
    tokenizer: PreTrainedTokenizerBase,
    languages: list[Language],
    save_filename: str | None = None
) -> None:
    fig = go.Figure()
    buttons = []

    for idx, token_translations in enumerate(translated_token_dataset):
        tokenized_text = tokenizer.tokenize(token_translations.text)
        tokenized_text = [decode_if_bytes(token) for token in tokenized_text]
        table = []

        for token, translations in zip(
            tokenized_text, token_translations.token_translations, strict=True
        ):
            if translations is None:
                row = [token] + [""] * len(languages)
            else:
                assert set(translations.keys()) == set(languages)
                row = [token] + tokenizer.batch_decode(
                    [translations[language] for language in languages]
                )
            table.append(row)

        table = list(map(list, zip(*table)))  # Transpose
        fig.add_trace(
            go.Table(
                header=dict(values=["original"] + languages),
                cells=dict(values=table),
                visible=False,  # Initially hide all tables
            )
        )

        # Create a button for each dataset
        button = dict(
            label=f"Prompt {idx+1}",
            method="update",
            args=[
                {"visible": [False] * len(translated_token_dataset)},
                {"title": f"{token_translations.text}<br>{tokenized_text}"},
            ],
        )
        button["args"][0]["visible"][idx] = True  # Show the selected table
        buttons.append(button)

    # Make the first table visible by default
    if fig.data:
        fig.data[0].visible = True

    # Add the dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=1.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        title=f"{translated_token_dataset[0].text}<br>{tokenizer.tokenize(translated_token_dataset[0].text)}",
    )

    save_or_show(fig, save_filename)
