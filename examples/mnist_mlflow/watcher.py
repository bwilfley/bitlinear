"""Per-layer weight and gradient statistics, logged to MLflow as scalar metrics.

Every quantity here is a scalar time series, which the MLflow UI plots and
compares across runs natively. Distribution histograms are a separate concern:
MLflow has no histogram widget, so those belong in an artifact plus an external
viewer rather than in this module.

Metric keys are shaped ``<kind>/<stat>/<param>`` -- e.g. ``grad/l2/fc1.weight``
-- so the MLflow UI groups them by kind.
"""

from contextlib import contextmanager

import mlflow
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

_STATS = ("mean", "std", "l2", "absmax", "abs_p50", "abs_p99", "frac_zero")


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
        "frac_zero": (flat == 0).float().mean().item(),
    }


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
    if module.weight_measure is None:
        return {}
    weight = module.weight.detach()
    w_scale = scale(weight, module.weight_range, module.weight_measure, False, module.eps)
    scaled = weight * w_scale
    limit = max(abs(bound) for bound in module.weight_range)

    stats = {f"scaled_{name}": value for name, value in _stats(scaled).items()}
    stats["w_scale"] = w_scale.item()
    stats["frac_saturated"] = (scaled.abs() > limit).float().mean().item()
    stats["frac_nonzero"] = _frac_nonzero(module, scaled)
    return stats


class ParamWatcher:
    """Samples per-parameter weight and gradient statistics into MLflow.

    Call :meth:`sample` once per training batch, after ``optimizer.step()`` and
    before the next ``zero_grad()``: ``step()`` leaves ``.grad`` populated, so a
    single call captures the post-update weights alongside the gradients that
    produced them, under one step index.
    """

    def __init__(self, model, interval=200, optimizer=None, eval_weights=False):
        self.model = model
        self.interval = interval
        self.optimizer = optimizer
        self.eval_weights = eval_weights
        self.bitlinears = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, BitLinear)
        }
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

    def sample(self, step):
        if self.interval <= 0 or step % self.interval:
            return

        metrics = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            for stat, value in _stats(parameter.grad).items():
                metrics[f"grad/{stat}/{name}"] = value

        with self._weight_view():
            for name, parameter in self.model.named_parameters():
                for stat, value in _stats(parameter).items():
                    metrics[f"weight/{stat}/{name}"] = value
            for name, module in self.bitlinears.items():
                for stat, value in _bitlinear_stats(module).items():
                    metrics[f"bitlinear/{stat}/{name}"] = value

        self._log(metrics, step)

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
