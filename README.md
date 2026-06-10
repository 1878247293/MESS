# MESS

MESS is a multi-table entity matching project built around sentence embeddings, automatic attribute selection, contrastive fine-tuning, and table merging.

The repository currently includes:

- the main matching pipeline
- a contrastive training entrypoint
- a Gradio web UI
- an LLM-based training-data generator
- bundled benchmark datasets

## Project Structure

```text
MESS/
|- data/                  # benchmark datasets
|- llm_data_generator/    # LLM-based labeled-pair generation
|- llm_training_data/     # generated labeled_pairs.json for supervised contrastive learning
|- src/                   # core implementation
|- web/                   # Gradio UI
|- main.py                # main entity matching pipeline
|- train_contrastive.py   # contrastive fine-tuning entrypoint
`- requirements.txt
```

## Requirements

- Python 3.10 or 3.11 is recommended
- CUDA is recommended for embedding and training, but CPU mode is also possible
- a local sentence-transformer model is expected by default

For this project, a local MiniLM model is enough. Prepare:

- `model/all-MiniLM-L12-v2`

If that folder does not exist, either:

1. prepare it locally under `model/`, or
2. override `--lm-model-or-path` when running

## Install

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the main pipeline on a dataset:

```bash
python main.py --data-name Geo --model-type minilm --lm-model-or-path model/all-MiniLM-L12-v2
```

Run contrastive fine-tuning:

```bash
python train_contrastive.py --data-name Geo --model-type minilm --lm-model-or-path model/all-MiniLM-L12-v2 --cl-epochs 10
```

This requires:

```text
llm_training_data/<dataset>/labeled_pairs.json
```

## Web UI

Launch the Gradio interface with:

```bash
python -m web.app
```

If your environment is set up correctly, the UI will open a local Gradio app for:

- main matching runs
- contrastive-learning runs
- LLM training-data generation
- result browsing

## Datasets

Bundled datasets are under `data/`:

- `Geo`
- `Music-20`
- `Music-200`
- `Music-2000`
- `Person`
- `Shopee`

Each dataset directory is expected to contain `table_*.csv` files and `ground_truth.txt`.

## LLM Training Data

Supervised contrastive learning reads labeled data from:

```text
llm_training_data/<dataset>/labeled_pairs.json
```

The current repository does not bundle these JSON files by default after cleanup.  
If you want to use supervised contrastive learning, generate them first.

To generate or refresh LLM-based training data:

```bash
python -m llm_data_generator.main --dataset Geo --data-dir data --output-dir llm_training_data
```

## Outputs

The project creates output directories during runtime, including:

- `results/`
- `logs/`
- `cache/`
- `training_logs/`
- `candidates_output/`

These are generated artifacts and usually should not be committed.

## Notes

- `main.py` is the primary entrypoint.
- The repository assumes offline/local model loading by default.
- For this project, MiniLM is enough; you do not need to prepare a separate local BERT-family model beyond `model/all-MiniLM-L12-v2`.

## Suggested First Run

If you only want to verify the project boots correctly, start with:

```bash
python main.py --data-name Geo --model-type minilm --lm-model-or-path model/all-MiniLM-L12-v2 --device cpu
```

Then switch to your local CUDA device after confirming the environment is healthy.
