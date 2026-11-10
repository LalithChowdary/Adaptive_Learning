"""Batch learner-skill records and train recurrent models on the selected device."""
from dataclasses import dataclass
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from .evaluation import validation_losses
from .features import TASK_SIZE, TOKEN_SIZE
from .recurrent import LearnerGRU


def training_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to verify native GPU training.")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple GPU access is unavailable to PyTorch MPS in this process. "
            "Run from macOS Terminal with the arm64 environment and GPU access. "
            "Use --device cpu only when intentionally running a CPU check."
        )
    device = torch.device("mps")
    probe = torch.ones(32, device=device, requires_grad=True)
    probe.square().mean().backward()
    torch.mps.synchronize()
    if probe.grad is None or probe.grad.device.type != "mps":
        raise RuntimeError("MPS backward pass did not produce GPU gradients")
    return device


@dataclass(frozen=True)
class Batch:
    inputs: tuple
    labels: torch.Tensor
    valid: torch.Tensor
    row_indices: np.ndarray
    count: int


def make_batches(partition, batch_size, device):
    sequences = partition.sequences
    lengths = [len(sequence.tokens) for sequence in sequences]
    order = np.argsort(lengths, kind="stable")
    offsets = np.cumsum([0, *(len(sequence.labels) for sequence in sequences)])
    batches = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        selected = [sequences[index] for index in indices]
        n = len(selected)
        steps = max(len(sequence.tokens) for sequence in selected)
        queries = max(len(sequence.labels) for sequence in selected)
        tokens = np.zeros((n, steps, TOKEN_SIZE), dtype=np.float32)
        gaps = np.zeros((n, steps), dtype=np.float32)
        positions = np.zeros((n, queries), dtype=np.int64)
        tasks = np.zeros((n, queries, TASK_SIZE), dtype=np.float32)
        query_gaps = np.zeros((n, queries), dtype=np.float32)
        labels = np.zeros((n, queries), dtype=np.float32)
        valid = np.zeros((n, queries), dtype=bool)
        rows = []
        for row, (index, sequence) in enumerate(zip(indices, selected)):
            length, count = len(sequence.tokens), len(sequence.labels)
            tokens[row, :length] = sequence.tokens
            gaps[row, :length] = sequence.gaps
            positions[row, :count] = sequence.positions
            tasks[row, :count] = sequence.tasks
            query_gaps[row, :count] = sequence.query_gaps
            labels[row, :count] = sequence.labels
            valid[row, :count] = True
            rows.extend(range(offsets[index], offsets[index + 1]))
        inputs = tuple(torch.from_numpy(x).to(device) for x in (tokens, gaps, positions, tasks, query_gaps))
        batches.append(Batch(
            inputs, torch.from_numpy(labels).to(device), torch.from_numpy(valid).to(device),
            np.asarray(rows), len(rows),
        ))
    return batches


@torch.no_grad()
def predict_recurrent(model, batches, query_count):
    model.eval()
    predictions = np.empty(query_count, dtype=np.float64)
    for batch in batches:
        logits = model(*batch.inputs)
        probabilities = logits.sigmoid().cpu().numpy()
        predictions[batch.row_indices] = probabilities[batch.valid.cpu().numpy()]
    return predictions


def fit_recurrent(
    mode, training, validation, validation_batches, masks, device, *, epochs, seed, scheduled=False,
):
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LearnerGRU(mode=mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001, foreach=False)
    scheduler = None
    if scheduled:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003, epochs=epochs, steps_per_epoch=len(training),
            pct_start=0.15, cycle_momentum=False,
        )
    curve = []
    best_loss, best_state, best_epoch = float("inf"), None, 0
    verified_gradient = None

    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        total = torch.zeros((), device=device)
        count = 0
        for index in rng.permutation(len(training)):
            batch = training[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch.inputs)
            per_query = F.binary_cross_entropy_with_logits(logits, batch.labels, reduction="none")
            loss_sum = (per_query * batch.valid).sum()
            (loss_sum / batch.count).backward()
            if verified_gradient is None:
                gradient = model.cell.weight_hh.grad
                if gradient is None or gradient.device.type != device.type or not gradient.isfinite().all().item():
                    raise RuntimeError("Recurrent backward pass did not produce finite gradients on the requested device")
                verified_gradient = str(gradient.device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss_sum.detach()
            count += batch.count

        training_loss = total.item() / count
        if not math.isfinite(training_loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        predictions = predict_recurrent(model, validation_batches, len(validation.labels))
        losses = validation_losses(validation.labels, predictions, masks, balanced=scheduled)
        score = losses["validation_selection_log_loss"]
        step = {
            "epoch": epoch, "train_log_loss": training_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **losses,
            "seconds": round(time.monotonic() - started, 3),
        }
        curve.append(step)
        print(f"  {mode} epoch {epoch}: train={training_loss:.5f} val selection={score:.5f} ({step['seconds']:.1f}s)", flush=True)
        if score < best_loss:
            best_loss, best_epoch = score, epoch
            best_metrics = losses
            best_state = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
        patience = 8 if scheduled else 3
        warmdown = not scheduled or epoch >= math.ceil(0.75 * epochs)
        if warmdown and epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "device": str(next(model.parameters()).device), "verified_gradient_device": verified_gradient,
        "hidden_size": model.hidden_size, "best_epoch": best_epoch,
        "seed": seed, "optimizer_steps": len(curve) * len(training),
        **best_metrics, "learning_curve": curve,
    }
