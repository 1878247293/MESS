"""
Contrastive learning training — Sentence-BERT fine-tuning.

Implements the SimCLR approach: InfoNCE + in-batch negative sampling.
Reads ready-made positive pairs from labeled_pairs.json as the supervision signal.
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file as load_safetensors
from pathlib import Path
from tqdm import tqdm


def _load_cached_weights(model: SentenceTransformer, cache_model_path: Path) -> SentenceTransformer:
    """
    Reuse an already-instantiated SentenceTransformer, only loading the cached weights into it.
    Avoids going through SentenceTransformer(path) again, whose loading path is relatively slow.
    """
    weights_path = cache_model_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_safetensors(str(weights_path), device=str(model.device))
        model[0].auto_model.load_state_dict(state_dict)
    else:
        # fallback: older versions may have saved a .bin
        weights_path = cache_model_path / "pytorch_model.bin"
        state_dict = torch.load(str(weights_path), map_location=model.device)
        model[0].auto_model.load_state_dict(state_dict)
    return model


# datasets

class SupervisedPairDataset(Dataset):
    """Read positive pairs from the labeled file"""

    def __init__(self, data_path: str, data_name: str):
        json_file = Path(data_path) / data_name / "labeled_pairs.json"
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        # support both historical formats
        # the new format is directly pairs (all positive), the old one is triplets, filtered by label==1
        if "pairs" in data:
            self.pairs = [(t[0], t[1]) for t in data["pairs"]]
        else:
            self.pairs = [(t[0], t[1]) for t in data["triplets"] if t[2] == 1]
        print(f"[SupervisedPairDataset] loaded {len(self.pairs)} positive pairs (from {json_file})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]  # (text_a, text_b)


class ContrastiveLearner:
    """Training wrapper that bundles model / loss / multi-GPU / logging together"""

    def __init__(self,
                 base_model: SentenceTransformer,
                 temperature: float = 0.07,
                 device: str = 'cuda',
                 multi_gpu: bool = False):
        self.model = base_model
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        # multi-GPU: wrap the underlying transformer with DataParallel
        self._using_data_parallel = False
        if multi_gpu and self.device.startswith('cuda') and torch.cuda.device_count() > 1:
            transformer = self.model._first_module()
            original_model = transformer.auto_model
            transformer.auto_model = torch.nn.DataParallel(original_model)
            transformer.auto_model.config = original_model.config
            self._using_data_parallel = True

        print(f"[ContrastiveLearner] init")
        print(f"  - model: {self.model._first_module().__class__.__name__}")
        print(f"  - device: {self.device}")
        if self._using_data_parallel:
            print(f"  - multi-GPU: DataParallel ({torch.cuda.device_count()} GPUs)")
        print(f"  - temperature: {self.temperature}")

    def _unwrap_data_parallel(self):
        """Restore the model after training; a DataParallel-wrapped model cannot be saved directly"""
        if self._using_data_parallel:
            transformer = self.model._first_module()
            transformer.auto_model = transformer.auto_model.module
            self._using_data_parallel = False

    def _save_loss_log(self, loss_history: list, epochs: int, data_name: str = "unknown"):
        """Write each epoch's loss to a JSON file, for plotting curves later"""
        log_dir = Path("training_logs")
        log_dir.mkdir(exist_ok=True)
        log_data = {
            "dataset": data_name,
            "epochs": list(range(1, epochs + 1)),
            "losses": [round(l, 6) for l in loss_history],
        }
        log_path = log_dir / f"{data_name}_loss.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"  loss log: {log_path}")

    def compute_contrastive_loss(self,
                                  embeddings1: torch.Tensor,
                                  embeddings2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE, computed in both directions and averaged.

        L = -log( exp(sim(z_i, z_i+) / tau) / sum_j exp(sim(z_i, z_j) / tau) )
        """
        batch_size = embeddings1.shape[0]

        # normalize, then use cosine similarity
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)

        # [B, B] similarity matrix, positive pairs on the diagonal
        similarity_matrix = torch.mm(embeddings1, embeddings2.t()) / self.temperature

        labels = torch.arange(batch_size, device=self.device)

#          exp([8.0, 2.0, 1.0]) = [2980.96, 7.39, 2.72]
#          sum             = 2991.07
#          Z_0 = log(2991.07) ~= 8.0036
#          nll_0 = Z_0 - S[0,0] = 8.0036 - 8.0 = 0.0036  <- very small loss; the positive pair scores far higher than the rest
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.t(), labels)

        loss = (loss_12 + loss_21) / 2

        return loss

    def train_epoch(self,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)  # progress-bar wrapper

        for text_a_batch, text_b_batch in pbar:
            # gradients must be retained, so we cannot use model.encode directly
            features_a = self.model.tokenize(text_a_batch)  # split text into token IDs, kept for the forward pass (gradients can flow)
            features_b = self.model.tokenize(text_b_batch)
            # move from CPU to GPU
            features_a = {k: v.to(self.device) for k, v in features_a.items()}
            features_b = {k: v.to(self.device) for k, v in features_b.items()}

            # forward pass to get sentence_embedding
            embeddings_a = self.model(features_a)['sentence_embedding']
            embeddings_b = self.model(features_b)['sentence_embedding']

            loss = self.compute_contrastive_loss(embeddings_a, embeddings_b)
            # clear gradients accumulated from the previous step
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # the optimizer updates weights according to the gradients
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self,
              data_path: str,
              data_name: str,
              epochs: int = 10,
              batch_size: int = 64,
              learning_rate: float = 1e-5,
              warmup_steps: int = 100) -> SentenceTransformer:
        """Train on positive pairs read from labeled_pairs.json"""
        print("\n" + "=" * 60)
        print("supervised CL")
        print("=" * 60)
        # read positive pairs
        dataset = SupervisedPairDataset(data_path, data_name)

        print(f"config:")
        print(f"  - num positive pairs: {len(dataset)}")
        print(f"  - epochs: {epochs}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - lr: {learning_rate}")
        print(f"  - warmup_steps: {warmup_steps}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps
            else max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
        )

        print(f"\nstarting training (total steps = {total_steps})...\n")

        best_loss = float('inf')
        loss_history = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, optimizer)
            loss_history.append(avg_loss)

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  -> new best loss: {best_loss:.4f}")

        self._save_loss_log(loss_history, epochs, data_name=data_name)

        print("\n" + "=" * 60)
        print("done")
        print(f"best loss: {best_loss:.4f}")
        print("=" * 60 + "\n")

        self._unwrap_data_parallel()
        return self.model


def contrastive_finetune(model: SentenceTransformer,
                         args,
                         cache_dir: str = "finetuned_models") -> SentenceTransformer:
    """
    Entry point for supervised contrastive learning. Labeled positive pairs are read from
    cl_training_data_dir/{data_name}/labeled_pairs.json.
    On a cache hit, load the weights directly; on a miss, train and then save.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # the cache name includes dataset / model type / key hyperparameters; keep the _supervised suffix for compatibility with historical cache directories
    model_type = getattr(args, 'model_type', 'minilm')
    if model_type == "minilm":
        cache_filename = f"{args.data_name}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}_supervised"
    else:
        cache_filename = f"{args.data_name}_{model_type}_cl_epochs{args.cl_epochs}_lr{args.cl_learning_rate}_temp{args.cl_temperature}_supervised"
    cache_model_path = cache_path / cache_filename

    if cache_model_path.exists() and not args.force_retrain:
        print("\n" + "=" * 60)
        print("Cache hit on fine-tuned model")
        print("=" * 60)
        print(f"  path: {cache_model_path}")
        print(f"  dataset: {args.data_name}")
        print(f"  skipping training, loading directly...")
        print("=" * 60 + "\n")

        try:
            _load_cached_weights(model, cache_model_path)
            print(f"loaded successfully: {cache_filename}\n")
            return model
        except Exception as e:
            print(f"failed to load: {e}")
            print("falling back to the training flow...\n")

    print("\n" + "=" * 60)
    print("No cache, starting training...")
    print("=" * 60 + "\n")

    learner = ContrastiveLearner(
        base_model=model,
        temperature=args.cl_temperature,
        device=args.device,
        multi_gpu=getattr(args, 'multi_gpu', False)
    )

    finetuned_model = learner.train(
        data_path=args.cl_training_data_dir,
        data_name=args.data_name,
        epochs=args.cl_epochs,
        batch_size=args.cl_batch_size,
        learning_rate=args.cl_learning_rate,
    )

    print("\n" + "=" * 60)
    print("Writing cache...")
    print("=" * 60)
    try:
        finetuned_model.save(str(cache_model_path))
        print(f"saved: {cache_model_path}")
        print(f"  the next run with the same parameters will use the cache directly")
    except Exception as e:
        print(f"save failed: {e}")
    print("=" * 60 + "\n")

    return finetuned_model
