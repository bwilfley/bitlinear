# Usage

Run all commands from `examples/mnist_array_metrics/`.

This example extends `../mnist_mlflow` with **binned weight and gradient
distributions**, written as a Parquet artifact and plotted outside the MLflow UI.
If all you want is scalar tracking, use `../mnist_mlflow` instead -- it is the
same trainer without the histogram and notebook machinery, and its dependency
list stays short as a result.

## Start the MLflow tracking server

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow_data/mlflow.db \
  --default-artifact-root mlflow_data/artifacts \
  --host 0.0.0.0 \
  --port 5002 \
  --disable-security-middleware
```

Port 5002, not 5001, so this example and `../mnist_mlflow` can run side by side
without sharing a database. (Point `--mlflow-uri` at 5001 if you would rather
keep everything in one place; the experiment names differ, so runs stay
distinct either way.)

## Run training

```bash
uv run python train.py --no-mps --watch --histograms --lr 1e-3
```

`--watch` logs the scalar metrics described in `../mnist_mlflow/USAGE.md`.
`--histograms` adds the distribution artifact. Both sample every
`--watch-interval` batches, defaulting to `--log-interval`.

Note the learning rate: at the default `--lr 0.1` a `--no-bitlinear` run collapses
to 11.35% within 200 batches. See `../mnist_mlflow/USAGE.md` for that failure and
what the metrics say about it.

## What lands where

The two tiers are complementary and nothing is stored twice.

| | Where | Answers |
|---|---|---|
| Scalar summaries | MLflow metrics | "is this layer's gradient vanishing?" |
| Binned histograms | Parquet artifact, `histograms/` | "what shape is it, and where did the mass go?" |

Scalars stay in MLflow because the UI already plots and cross-compares them.
Histograms cannot: MLflow has no histogram widget, so they are logged as data and
plotted separately.

### The histogram schema

One row per **occupied** bin -- empty bins are omitted, so an absent
`(step, param, kind, bin)` means a count of zero.

| Column | Notes |
|---|---|
| `run_id`, `step`, `epoch` | `step` is the global batch index |
| `param` | e.g. `fc1.weight` |
| `module_type` | `BitLinear`, `Conv2d`, ... |
| `kind` | `weight`, `grad`, or `scaled_weight` |
| `bin` | signed bin id; `0` is exact zero |
| `lo`, `hi` | signed bin bounds, denormalized onto every row |
| `count` | elements in the bin |

`scaled_weight` exists only for BitLinear layers and holds `weight * w_scale` --
the latent weights in the coordinates the quantizer actually sees. It is the
interesting distribution: whether a ternary layer is saturating or collapsing to
zero is a statement about where this sits relative to ±1, not about the raw
weights.

**Binning.** Fixed signed-log2 bins, two per octave, spanning `2**-32` to
`2**16`, plus a dedicated bin for exact zero. The edges are constant for the
whole run, which is the point: every row is directly comparable and rows from
different runs concatenate with no edge reconciliation. That is the bookkeeping
that per-tensor adaptive binning (what `wandb.watch` does) pushes onto the
reader. Exact zero gets its own bin because a zero gradient means a dead unit,
while a merely tiny one does not.

Two consequences for whoever plots this:

- The outermost bins are unbounded, so `lo`/`hi` can hold `±inf`. Clamp or drop
  them before handing the data to a plotting library.
- `lo` and `hi` are signed and the natural axis is symmetric-log. Vega-Lite's
  `symlog` scale handles this directly.

## Cost

Measured on Apple Silicon, 12 CPU threads, against a 58 ms training batch:

| | Per sample | Overhead at `--watch-interval 200` |
|---|---|---|
| `--watch` | 71 ms | 0.61% |
| `--watch --histograms` | 108 ms | 0.93% |

A 14-epoch run samples 66 times and produces roughly 32k rows, a few hundred KB
of zstd-compressed Parquet. Shards are flushed and uploaded as the run proceeds,
so a run that dies keeps whatever it already sampled.

Neither flag perturbs the run. Statistics are computed without consuming global
RNG -- which matters here, because the `sample` strategy draws from it on every
forward pass -- so a watched run's trajectory is bit-identical to an unwatched one
at the same seed. Verified for `--watch` and `--watch --histograms` alike.

## Plotting the distributions

```bash
uv run marimo edit notebooks/distributions.py
```

The notebook reads shapes from the Parquet artifact and scalars from the tracking
server, and offers five views:

| View | Shows |
|---|---|
| Distribution over training | `step` x symmetric-log value, coloured by share of the tensor |
| Magnitude envelope | the span holding the middle 98% of the mass, over time |
| Latent weights vs the grid | `scaled_weight` with the ±1 ternary boundary marked |
| Saturation | share of latent weights past ±1, per layer |
| Scalar metrics | the tier-1 series, straight from MLflow |

Pick runs with the multiselect at the top to overlay them; only runs logged with
`--histograms` appear. Filtering happens in polars before anything reaches
Altair, which keeps each chart under Altair's inline-data limit without needing a
custom data transformer.

Two conventions the notebook applies, worth knowing if you write your own views:

- **Bin 0 (exact zero) is excluded from the shape charts.** It has zero width, so
  a rect mark would render it invisible. Read it as `frac_zero` in the scalar
  view instead -- that is the dead-unit signal, and it deserves a line rather
  than a stripe.
- **Counts are normalized within each `(run, step)`.** Raw counts mostly restate
  tensor size, which makes layers and runs incomparable.

To sanity-check any change to the sampling code, note that the saturation view is
derivable two ways: `1.0` is exactly a bin edge (`2**0`), so summing the
histogram past it should reproduce `bitlinear/frac_saturated` to within the one
element that a left-closed bin and a strict `>` classify differently.

## Reading the artifact

```python
import mlflow, pyarrow.parquet as pq

path = mlflow.artifacts.download_artifacts(run_id=RUN_ID, artifact_path="histograms")
table = pq.read_table(path)          # shards concatenate automatically
```

The histogram is independently checkable against the scalar metrics: bin 0's
share of the total equals `<kind>/frac_zero/<param>` exactly, and the occupied
bins bracket `<kind>/absmax/<param>`. Those two agreeing is a useful smoke test
after any change to the sampling code.

## View results

Open **http://localhost:5002** for the scalar metrics. The histograms are on each
run under the `histograms/` artifact path.
