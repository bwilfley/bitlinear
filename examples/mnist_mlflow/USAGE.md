# Usage

Run all commands from `examples/mnist_mlflow/`.

## Start the MLflow tracking server

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow_data/mlflow.db \
  --default-artifact-root mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5001 \
  --disable-security-middleware
```

The server persists state in `mlflow_data/` (git-ignored). Port 5001 avoids
the macOS AirPlay Receiver, which occupies port 5000.

## Run training

```bash
uv run python train.py --no-mps
```

`--no-mps` runs on CPU. Common options:

| Flag | Default | Description |
|---|---|---|
| `--epochs N` | 14 | Number of training epochs |
| `--lr LR` | 0.1 | Learning rate |
| `--batch-size N` | 64 | Training batch size |
| `--no-bitlinear` | off | Use standard Linear layers instead of BitLinear |
| `--save-model` | off | Save model weights locally and to MLflow |
| `--dry-run` | off | One batch per epoch (quick smoke test) |
| `--watch` | off | Log per-layer weight and gradient statistics |

## Watching weights and gradients

```bash
uv run python train.py --no-mps --watch
```

This logs ~135 scalar metric series describing each layer's weight and gradient
distributions, sampled every `--watch-interval` batches (defaults to
`--log-interval`). They are ordinary MLflow metrics, so the UI plots them over
time and compares them across runs with no extra tooling.

| Flag | Default | Description |
|---|---|---|
| `--watch-interval N` | `--log-interval` | Batches between samples |
| `--watch-eval-weights` | off | Read the schedule-free averaged weights instead of the training iterate |

Keys are shaped `<kind>/<stat>/<param>`, e.g. `grad/l2/fc1.weight`:

- **`grad/…`** and **`weight/…`** — `mean`, `std`, `l2`, `absmax`, `abs_p50`,
  `abs_p99`, `frac_zero` for every parameter. `grad/l2` and `grad/abs_p99` are
  the vanishing/exploding indicators.
- **`bitlinear/…`** — per BitLinear layer: `w_scale`, `frac_nonzero` (the
  fraction of weights quantizing to ±1 rather than 0), and the same seven stats
  over the *scaled* latent weights `weight * w_scale`, prefixed `scaled_`. The
  scaled distribution is the one that matters: it is what the quantizer sees.

Two caveats worth knowing:

- With the default `AbsMedian` weight measure, `w_scale` is by definition
  `1 / median(|w|)`, which pins `bitlinear/scaled_abs_p50` to 1.0 and
  `bitlinear/frac_saturated` to 0.5 no matter what training does. Ignore them
  here — but `scaled_abs_p50` is useful backwards: because its true value is
  known to be exactly 1.0, its deviation measures the error of the percentile
  subsampling described below.
- Weights are read at the schedule-free optimizer's *training* iterate by
  default, which is not the averaged point that produces the reported accuracy.
  `--watch-eval-weights` reads the averaged point instead; gradients are the same
  either way. The run records which was used as the `watch_weight_point` param.

Watching does not perturb the run: statistics are computed without consuming
global RNG, so a watched run's trajectory is bit-identical to an unwatched one at
the same seed.

Cost is roughly 54 ms per sample on CPU against an 80 ms training batch — about
0.3% at the default interval of 200, but ~70% if you drop the interval to 1.
Sample sparsely unless you are chasing something specific.

The `abs_p50` and `abs_p99` metrics are estimated from a 131k-element subsample
of any larger tensor; every other statistic is exact. Subsampling error depends
on how heavy-tailed the distribution is — about 0.03% on freshly initialized
weights, but up to ~2% once BitLinear's latent weights spread out during training
(measured against the analytically pinned `scaled_abs_p50`). That is well inside
the tolerance for reading trends, which is what these metrics are for. Raise
`_QUANTILE_MAX` in `watcher.py` if you need tighter: error falls as `1/sqrt(n)`,
so halving it costs roughly 4x the sampling time.

### Worked example: BitLinear

The command at the top of this section, run to completion on all 14 epochs
(Apple Silicon, 12 CPU threads): 853 s wall clock, 99.11% test accuracy, 66
watcher samples across 13,000 batches, 135 metric series. Two things it surfaced
which the loss curve does not show:

- **`grad/frac_zero/conv2.weight` jumps to 0.46 within the first epoch and
  drifts up to ~0.51.** That is not scattered sparsity: at the 0.4614 sample,
  28 of conv2's 64 output filters have an identically zero gradient, and the
  remainder of the fraction is scatter inside the live filters. Nearly half the
  layer is dead while the model still reaches 99.11%. `conv1.weight` settles at
  exactly 0.03125 = 1/32 late in training, consistent with a single dead filter.
- **`bitlinear/scaled_absmax/fc1` climbs from 1.1 to 90 while `w_scale` falls
  from 10 to 2.1.** The latent weights are drifting far outside the ternary
  range: the top percentile ends up ~33x beyond the clamp ceiling, where the
  quantizer output is already saturated at ±1 and further growth changes nothing
  in the forward pass. Meanwhile `frac_nonzero` settles near 0.67, so the layer
  converges to roughly one third zeros. Much of this drift is learning rate
  rather than something inherent to BitLinear: at `--lr 1e-3` the same layer ends
  at `scaled_absmax` 14 and `scaled_abs_p99` 5.8, so the tail still escapes the
  ternary range but by ~6x rather than ~33x.

### Worked example: the standard-Linear baseline

```bash
uv run python train.py --no-mps --no-bitlinear --watch --lr 1e-3
```

Note the learning rate. At the default `--lr 0.1` this baseline does not train at
all — it scores 11.35%, and the watcher says why in one query: `grad/frac_zero`
is exactly **1.0 for every parameter** from batch 200 onward, except `fc2.bias`,
which keeps a healthy gradient. That is total dead-ReLU collapse. The first few
steps drive every ReLU negative for every input, so only the output bias can
still learn, and it converges to predicting the single most frequent class —
11.35% is exactly MNIST's 1135 test images of class 1. A flat 2.30 loss curve
(= ln 10, a uniform posterior) looks identical to "learning slowly"; the gradient
metrics distinguish the two immediately. That failure is worth reproducing once
to see what a dead network looks like in these plots.

At `--lr 1e-3` the baseline trains normally: 639 s, 99.31%. So BitLinear costs
about 33% more wall clock and lands 0.2 points lower here, but tolerates a 100x
higher learning rate than plain `Linear` survives.

### Reading the runs together

Comparing BitLinear at `lr 0.1` against the baseline at `lr 1e-3` confounds layer
type with learning rate, so it cannot support a causal claim either way. The
third run below is the matched pair that separates them:

```bash
uv run python train.py --no-mps --watch --lr 1e-3
```

`grad/frac_zero` at the final sample (step 13000). The first two columns are the
controlled comparison; the third shows how much of the effect is learning rate:

| Parameter | base, 1e-3 | BitLinear, 1e-3 | BitLinear, 0.1 |
|---|---|---|---|
| `conv1.weight` | 0.000 | 0.000 | 0.031 |
| `conv2.weight` | 0.052 | 0.024 | 0.505 |
| `fc1.weight` | 0.625 | **0.000** | 0.140 |
| `fc1.bias` | 0.172 | **0.000** | 0.117 |
| `fc2.weight` | 0.175 | **0.000** | 0.000 |

Accuracy: 99.31% (base 1e-3), 99.23% (BitLinear 1e-3), 99.11% (BitLinear 0.1).
Wall clock 639 s / 809 s / 853 s — BitLinear costs roughly 25-33% more per run.

**Every parameter of a BitLinear layer has exactly zero gradient sparsity**,
against 0.17-0.63 for the same layers as plain `Linear`. This is structural, not
a learning-rate artifact — it holds at both learning rates and is *cleaner* at
`1e-3`. `BitLinear.forward` applies `LayerNorm` to its input before quantizing,
and mean-subtraction turns exact zeros — dead ReLU outputs, plus the 25% that
`dropout1` zeroes — into nonzero values, so `grad[i,j] = dL/dy_i * x_j` stops
vanishing. The quantizer itself does not kill gradient either: both `round_clamp`
and `sample` are straight-through (`(q - x).detach() + x`, identity gradient),
and `scale()` detaches its input.

**BitLinear does not cause conv2's dead filters.** The obvious reading of the
first two runs — that BitLinear drives `conv2.weight` sparsity 10x up, from 0.052
to 0.505 — does not survive control. At matched learning rate BitLinear scores
0.024 against the baseline's 0.052, i.e. slightly *fewer* dead filters. The
0.505 is entirely an artifact of `lr 0.1`, which kills ReLUs on its own. This is
the trap these metrics invite: layer-type differences and learning-rate
differences produce similar-looking curves, and only a matched pair tells them
apart.

## View results

Open **http://localhost:5001** in a browser. Runs appear under the
`mnist-bitlinear` experiment as they log.
