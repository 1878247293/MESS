"""
Attribute selection (EER, Embedding Equivalence Removal).

The first stage of the main pipeline. Goal: remove columns that do not affect entity identity
(such as random IDs), keeping only the attributes that truly distinguish entities, to reduce
noise in the subsequent KNN search.

Method: for each candidate column, shuffle the entire column, re-encode the whole table, and
compute a row-wise cosine against the original encoding. If the mean similarity
`mean_sim <= col_sim_threshold`, the row vectors change a lot when the column is shuffled, so the
column matters for identity and is kept; otherwise it is discarded. The result can be cached and
reused by selector_cache.
"""

from typing import List
import os

# turn on the offline switches before importing HF / SentenceTransformers
# otherwise the attribute-selection stage occasionally probes the network silently
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from args import MainArgs
from data import textify_table
from log import log
from utils import element_wise_cosine_sim


def auto_selection(tables_df: List[pd.DataFrame], args: MainArgs):
    table_df = pd.concat(tables_df, axis=0)  # concatenate tables vertically
    table_df = table_df.sample(frac=args.selection_rate)

    model = SentenceTransformer(
        args.lm_model_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.max_seq_length = args.max_seq_length
    model.to(args.device)
    # join each DataFrame row into one sentence
    sentences_before = textify_table(table_df)
    table_embeddings = model.encode(
        sentences_before,
        show_progress_bar=True,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )

    selected_attrs = ["tid"]
    attribute_scores = {}

    for name, col in table_df.items():
        if name == "tid":
            continue

        col_copy = col.copy(deep=True)
        table_df[name] = col_copy.sample(frac=1).reset_index(drop=True)
        sentences_after = textify_table(table_df)
        table_df[name] = col

        table_embeddings_after = model.encode(
            sentences_after,
            show_progress_bar=True,
            batch_size=args.batch_size,
            normalize_embeddings=True,
        )
        sim = element_wise_cosine_sim(table_embeddings, table_embeddings_after)  # original vector a . shuffled vector b
        mean_sim = np.mean(sim)  # take the mean
        attribute_scores[name] = float(mean_sim)

        if mean_sim <= args.col_sim_threshold:
            selected_attrs.append(name)

        log(f"col: {name}, sim: {mean_sim}")

    return selected_attrs, attribute_scores
