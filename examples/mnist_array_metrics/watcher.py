"""Per-layer weight and gradient distributions, sampled during training.

Two things come out of one pass over the parameters, split by what the MLflow UI
is actually good at:

* **Scalar summaries** go to MLflow as ordinary metrics, keyed
  ``<kind>/<stat>/<param>`` (e.g. ``grad/l2/fc1.weight``). The UI plots them over
  time and compares them across runs with no extra tooling.
* **Binned histograms** go to a Parquet artifact on the run. MLflow has no
  histogram widget, so these are data rather than a visual; plot them with
  ``notebooks/distributions.py``.

The two are complementary, not redundant. The scalars answer "is this layer's
gradient vanishing"; the histograms answer "what shape is it, and where did the
mass go". Scalars are cheap enough to sample often, histograms are not much
worse, and neither is stored twice -- nothing in the Parquet duplicates a metric.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager

import mlflow
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# `scale` is not re-exported from bitlinear/__init__.py, but reimplementing it
# here would drift from BitLinear.forward, which is the thing being measured.
from bitlinear.bitlinear import BitLinear, scale

# torch.quantile sorts its input, which dominates the cost of watching a wide
# layer -- an unsubsampled 1.2M-element fc1.weight costs ~200ms, versus ~16ms to
# ship every metric in this module to the tracking server. Percentiles estimated
# from 131k stride-subsamples are accurate to far better than these metrics are
# read, and the sample is deterministic on purpose: drawing random indices would
# advance the global RNG and so perturb the very run this module observes. (The
# cap also stays clear of torch.quantile's own ~16M element ceiling.)
_QUANTILE_MAX = 1 << 17

# Asking for both quantiles at once costs one sort rather than two.
_QUANTILES = torch.tensor([0.5, 0.99])

# Fixed signed-log2 bins, two per octave. Holding the edges constant for the
# whole run is the point: every sampled row is then directly comparable, and rows
# concatenate across runs with no edge reconciliation -- which is the bookkeeping
# that per-tensor adaptive binning (what wandb.watch does) forces on the reader.
# Gradients span decades, so log spacing is the only sane choice, and 2**-32 to
# 2**16 brackets both vanishing and exploding without truncating either. Only
# occupied bins are written, so carrying a wide unused range costs nothing.
_OCTAVE_LO, _OCTAVE_HI, _BINS_PER_OCTAVE = -32, 16, 2
_EDGES = torch.tensor([
    2.0 ** (exponent / _BINS_PER_OCTAVE)
    for exponent in range(_OCTAVE_LO * _BINS_PER_OCTAVE, _OCTAVE_HI * _BINS_PER_OCTAVE + 1)
])

# One row per occupied bin. `lo`/`hi` are denormalized onto every row so the
# reader never has to join against an edge table; the outer two bins are
# unbounded, so those columns can hold +/-inf and a plot must clamp them.
_HIST_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("step", pa.int32()),
    ("epoch", pa.int16()),
    ("param", pa.string()),
    ("module_type", pa.string()),
    ("kind", pa.string()),
    ("bin", pa.int16()),
    ("lo", pa.float32()),
    ("hi", pa.float32()),
    ("count", pa.int64()),
])


def _stats(tensor):
    """Scalar summary of one tensor's distribution.

    Runs on CPU: torch.quantile has patchy MPS coverage, and copying a few MB at
    watch cadence costs nothing next to a training step.
    """
    flat = tensor.detach().flatten().float().cpu()
    n = flat.numel()
    if n == 0:
        return {}
    absolute = flat.abs()
    stride = n // _QUANTILE_MAX + 1
    sub = absolute if stride == 1 else absolute[::stride]
    p50, p99 = sub.quantile(_QUANTILES).tolist()
    return {
        # Everything but the percentiles is a single exact pass over the tensor.
        "mean": flat.mean().item(),
        "std": flat.std().item() if n > 1 else 0.0,
        "l2": flat.norm().item(),
        "absmax": absolute.max().item(),
        "abs_p50": p50,
        "abs_p99": p99,
        # Counted as an integer rather than a float32 mean, which loses ~1e-8 over
        # a million elements. That is irrelevant to the metric itself, but it lets
        # this agree exactly with the bin-0 count in the histogram, so any future
        # disagreement between the two is a real bug rather than rounding.
        "frac_zero": int((flat == 0).sum()) / n,
    }


def _histogram(tensor):
    """Sparse signed-log2 histogram as ``{bin_id: count}``, empty bins omitted.

    Bin 0 is exact zero, which deserves its own bin rather than being folded into
    the smallest magnitude bucket: a zero gradient means a dead unit, while a
    merely tiny one does not.  Remaining ids are ``+/-(bucket + 1)``, where
    bucket 0 underflows the edge table and the last bucket overflows it.
    """
    flat = tensor.detach().flatten().float().cpu()
    counts = {}

    zeros = flat == 0
    n_zero = int(zeros.sum())
    if n_zero:
        counts[0] = n_zero

    nonzero = flat[~zeros]
    if nonzero.numel():
        # right=True gives left-closed bins, [edges[i-1], edges[i]).
        bucket = torch.bucketize(nonzero.abs(), _EDGES, right=True)
        signed = torch.where(nonzero > 0, bucket + 1, -(bucket + 1))
        ids, occupancy = torch.unique(signed, return_counts=True)
        counts.update(zip(ids.tolist(), occupancy.tolist()))
    return counts


def _bin_bounds(bin_id):
    """Signed ``(lo, hi)`` interval for a bin id. The outer bins are unbounded."""
    if bin_id == 0:
        return 0.0, 0.0
    bucket = abs(bin_id) - 1
    lo = 0.0 if bucket == 0 else float(_EDGES[bucket - 1])
    hi = float("inf") if bucket >= len(_EDGES) else float(_EDGES[bucket])
    return (lo, hi) if bin_id > 0 else (-hi, -lo)


def _scaled_weight(module):
    """A BitLinear layer's latent weights in the coordinates the quantizer sees.

    Returns ``(scaled, w_scale)``, or ``(None, None)`` when the layer does not
    quantize its weights.
    """
    if module.weight_measure is None:
        return None, None
    weight = module.weight.detach()
    w_scale = scale(weight, module.weight_range, module.weight_measure, False, module.eps)
    return weight * w_scale, w_scale


def _frac_nonzero(module, scaled):
    """Fraction of weights that quantize to something other than zero.

    The `sample` strategy is stochastic, so calling it here would both consume
    global RNG and return a noisy answer. For ternary sampling the expectation is
    closed form: P(nonzero) = P(rand <= |x|) = clamp(|x|, 0, 1). Every other
    strategy is deterministic, so just run it -- with lambda_ pinned to 1.0, since
    the straight-through blend the module uses in training is not a quantized
    value and would not compare against zero meaningfully.
    """
    stochastic = getattr(module.strategy, "__name__", "") == "sample"
    if stochastic and tuple(module.weight_range) == (-1, 1):
        return scaled.abs().clamp(max=1.0).mean().item()
    quantized = module.strategy(scaled, module.weight_range, 1.0)
    return (quantized != 0).float().mean().item()


def _bitlinear_stats(module):
    """Diagnostics specific to the straight-through quantizer.

    What governs a BitLinear layer is not the magnitude of its latent weights but
    where they sit relative to ``w_scale``: that ratio is what the quantizer
    actually sees, and it decides the ternary occupancy. So the distribution of
    ``weight * w_scale`` is the informative one, not the distribution of
    ``weight``.

    Note that the measure pins some of these. Under the default ``AbsMedian`` and
    a ternary range, ``w_scale`` is by definition ``1 / median(|w|)``, so
    ``scaled_abs_p50`` is identically 1.0 and ``frac_saturated`` is identically
    0.5 -- flat lines that say nothing about training. They carry real signal
    under ``AbsMax`` or ``AbsMean``, so they stay, but read ``w_scale``,
    ``frac_nonzero``, ``scaled_abs_p99`` and ``scaled_std`` for the AbsMedian case.
    """
    scaled, w_scale = _scaled_weight(module)
    if scaled is None:
        return {}
    limit = max(abs(bound) for bound in module.weight_range)

    stats = {f"scaled_{name}": value for name, value in _stats(scaled).items()}
    stats["w_scale"] = w_scale.item()
    stats["frac_saturated"] = (scaled.abs() > limit).float().mean().item()
    stats["frac_nonzero"] = _frac_nonzero(module, scaled)
    return stats


class _ShardWriter:
    """Buffers histogram rows and flushes them as Parquet shards to the run.

    Shards rather than one file at the end, so a run that dies keeps the samples
    it already took. One ``log_artifact`` per shard rather than ``log_artifacts``
    over a directory, so nothing already uploaded is uploaded again.
    """

    def __init__(self, directory, artifact_path="histograms", flush_rows=200_000):
        self.directory = directory
        self.artifact_path = artifact_path
        self.flush_rows = flush_rows
        self.shard = 0
        self._columns = {name: [] for name in _HIST_SCHEMA.names}
        os.makedirs(directory, exist_ok=True)

    def __len__(self):
        return len(self._columns["step"])

    def add(self, **row):
        for name, value in row.items():
            self._columns[name].append(value)

    def maybe_flush(self):
        if len(self) >= self.flush_rows:
            self.flush()

    def flush(self):
        if not len(self):
            return
        path = os.path.join(self.directory, f"hist-{self.shard:05d}.parquet")
        pq.write_table(pa.table(self._columns, schema=_HIST_SCHEMA), path, compression="zstd")
        mlflow.log_artifact(path, artifact_path=self.artifact_path)
        self.shard += 1
        self._columns = {name: [] for name in _HIST_SCHEMA.names}


class ParamWatcher:
    """Samples per-parameter weight and gradient distributions into MLflow.

    Call :meth:`sample` once per training batch, after ``optimizer.step()`` and
    before the next ``zero_grad()``: ``step()`` leaves ``.grad`` populated, so a
    single call captures the post-update weights alongside the gradients that
    produced them, under one step index. Call :meth:`close` from a ``finally`` so
    a failed run still keeps the samples it managed to take.
    """

    def __init__(self, model, interval=200, optimizer=None, eval_weights=False,
                 histograms=False, histogram_dir=None):
        self.model = model
        self.interval = interval
        self.optimizer = optimizer
        self.eval_weights = eval_weights
        self.bitlinears = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, BitLinear)
        }
        self.module_types = {
            (f"{module_name}.{param_name}" if module_name else param_name): type(module).__name__
            for module_name, module in model.named_modules()
            for param_name, _ in module.named_parameters(recurse=False)
        }
        self._owned_dir = histogram_dir is None
        self.directory = histogram_dir or (tempfile.mkdtemp(prefix="watcher-") if histograms else None)
        self.writer = _ShardWriter(self.directory) if histograms else None
        self._warned = False

    @property
    def weight_point(self):
        """Which iterate :meth:`sample` reads weights from."""
        return "eval" if self.eval_weights and self.optimizer is not None else "train"

    @contextmanager
    def _weight_view(self):
        """Optionally swap in a schedule-free optimizer's averaged iterate.

        AdamWScheduleFree holds the parameters at an extrapolated point during
        training and only materializes the averaged weights under ``.eval()``.
        Those averaged weights are the ones that produce the reported accuracy,
        but gradients are always taken at the train point -- the two views answer
        different questions, so the caller chooses. Gradients are untouched by the
        swap either way.
        """
        swap = self.weight_point == "eval"
        if swap:
            self.optimizer.eval()
        try:
            yield
        finally:
            if swap:
                self.optimizer.train()

    def _record(self, tensor, step, epoch, param, kind):
        if self.writer is None or tensor is None:
            return
        run = mlflow.active_run()
        run_id = run.info.run_id if run else ""
        module_type = self.module_types.get(param, "")
        for bin_id, count in sorted(_histogram(tensor).items()):
            lo, hi = _bin_bounds(bin_id)
            self.writer.add(run_id=run_id, step=step, epoch=epoch, param=param,
                            module_type=module_type, kind=kind, bin=bin_id,
                            lo=lo, hi=hi, count=count)

    def sample(self, step, epoch=0):
        if self.interval <= 0 or step % self.interval:
            return

        metrics = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            for stat, value in _stats(parameter.grad).items():
                metrics[f"grad/{stat}/{name}"] = value
            self._record(parameter.grad, step, epoch, name, "grad")

        with self._weight_view():
            for name, parameter in self.model.named_parameters():
                for stat, value in _stats(parameter).items():
                    metrics[f"weight/{stat}/{name}"] = value
                self._record(parameter, step, epoch, name, "weight")
            for name, module in self.bitlinears.items():
                for stat, value in _bitlinear_stats(module).items():
                    metrics[f"bitlinear/{stat}/{name}"] = value
                scaled, _ = _scaled_weight(module)
                self._record(scaled, step, epoch, f"{name}.weight", "scaled_weight")

        self._log(metrics, step)
        if self.writer is not None:
            self.writer.maybe_flush()

    def close(self):
        """Flush the tail shard and drop the local staging copy."""
        if self.writer is None:
            return
        try:
            self.writer.flush()
        except Exception as error:
            print(f"warning: watcher failed to flush histograms: {error}")
        if self._owned_dir:
            shutil.rmtree(self.directory, ignore_errors=True)
        self.writer = None

    def _log(self, metrics, step):
        """Instrumentation must never take down the run it is instrumenting.

        An exploding gradient produces non-finite metrics, which the tracking
        backend may reject; losing the diagnostic is an acceptable outcome,
        losing the training run is not.
        """
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as error:
            if not self._warned:
                self._warned = True
                print(f"warning: watcher failed to log metrics at step {step}: {error}")
