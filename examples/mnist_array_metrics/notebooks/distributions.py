"""Interactive views of the weight and gradient distributions logged by train.py.

    uv run marimo edit notebooks/distributions.py

Scalars come from the MLflow tracking server and shapes come from the Parquet
artifact, each from exactly one place -- nothing here recomputes a number that
the other source already holds.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import functools

    import altair as alt
    import marimo as mo
    import mlflow
    import polars as pl

    return alt, functools, mlflow, mo, pl


@app.cell
def _(mo):
    tracking_uri = mo.ui.text(
        value="http://localhost:5002",
        label="Tracking URI",
        full_width=True,
    )
    tracking_uri
    return (tracking_uri,)


@app.cell
def _(mlflow, mo, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri.value)
    client = mlflow.MlflowClient()

    def _runs():
        found = []
        for experiment in client.search_experiments():
            for run in client.search_runs([experiment.experiment_id]):
                if run.data.params.get("histograms") == "True":
                    found.append(run)
        return sorted(found, key=lambda r: r.info.start_time, reverse=True)

    runs = _runs()
    labels = {
        f'{r.info.run_id[:8]}  bitlinear={r.data.params.get("bitlinear")} '
        f'lr={r.data.params.get("lr")} acc={r.data.metrics.get("accuracy", float("nan")):.2f}': r.info.run_id
        for r in runs
    }
    mo.stop(not labels, mo.md("**No runs with `--histograms` found.** Check the tracking URI."))

    run_picker = mo.ui.multiselect(
        options=labels,
        value=list(labels)[:1],
        label="Runs to compare",
    )
    run_picker
    return client, run_picker


@app.cell
def _(functools, mlflow, mo, pl, run_picker):
    @functools.lru_cache(maxsize=32)
    def _load(run_id):
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="histograms"
        )
        # Bins are sparse and the outer two are unbounded, so drop the infinite
        # edges before anything tries to put them on an axis. Bin 0 (exact zero)
        # has zero width and would be invisible in a rect mark -- it is carried
        # here but excluded from the shape charts and read as frac_zero instead.
        return (
            pl.scan_parquet(f"{path}/*.parquet")
            .filter(pl.col("lo").is_finite() & pl.col("hi").is_finite())
            .collect()
        )

    mo.stop(not run_picker.value, mo.md("Select at least one run."))
    frames = {run_id: _load(run_id) for run_id in run_picker.value}
    hist = pl.concat(frames.values())
    return (hist,)


@app.cell
def _(hist, mo):
    params = sorted(hist["param"].unique().to_list())
    kinds = sorted(hist["kind"].unique().to_list())

    param_picker = mo.ui.dropdown(
        options=params,
        value="fc1.weight" if "fc1.weight" in params else params[0],
        label="Parameter",
    )
    kind_picker = mo.ui.dropdown(
        options=kinds,
        value="grad" if "grad" in kinds else kinds[0],
        label="Kind",
    )
    mo.hstack([param_picker, kind_picker], justify="start", gap=2)
    return kind_picker, param_picker


@app.cell
def _(hist, kind_picker, param_picker, pl):
    # Normalize within each (run, step) so distributions are comparable over time
    # and between runs -- raw counts just restate the tensor size.
    selection = (
        hist.filter(
            (pl.col("param") == param_picker.value)
            & (pl.col("kind") == kind_picker.value)
            & (pl.col("bin") != 0)
        )
        .with_columns(
            fraction=pl.col("count") / pl.col("count").sum().over(["run_id", "step"])
        )
        .with_columns(run=pl.col("run_id").str.slice(0, 8))
    )
    return (selection,)


@app.cell
def _(mo):
    mo.md("""
    ## Distribution over training

    Each column is one sample; colour is the share of the tensor's elements in
    that bin. The y axis is symmetric-log, so the vertical spread *is* the
    order-of-magnitude spread of the values, and drift toward the bottom is
    exactly what a vanishing gradient looks like.
    """)
    return


@app.cell
def _(alt, mo, selection):
    heatmap = (
        alt.Chart(selection)
        .mark_rect()
        .encode(
            x=alt.X("step:O", title="step", axis=alt.Axis(labelOverlap=True)),
            y=alt.Y("lo:Q", title="value", scale=alt.Scale(type="symlog", constant=1e-8)),
            y2="hi:Q",
            color=alt.Color(
                "fraction:Q",
                title="share",
                scale=alt.Scale(type="log", scheme="viridis"),
            ),
            tooltip=["run:N", "step:Q", "lo:Q", "hi:Q", "count:Q", "fraction:Q"],
        )
        .properties(height=320, width=520)
        .facet(row=alt.Row("run:N", title=None))
        .resolve_scale(color="shared")
    )
    mo.ui.altair_chart(heatmap)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Magnitude envelope

    The same data collapsed to a magnitude profile: the band spans the bins
    holding the middle 98% of the mass, so a widening band means the tensor is
    spreading across more orders of magnitude.
    """)
    return


@app.cell
def _(alt, mo, pl, selection):
    envelope = (
        selection.with_columns(magnitude=pl.max_horizontal(pl.col("lo").abs(), pl.col("hi").abs()))
        .sort("magnitude")
        .with_columns(
            cumulative=pl.col("fraction").cum_sum().over(["run_id", "step"])
        )
        .filter((pl.col("cumulative") > 0.01) & (pl.col("cumulative") < 0.99))
        .group_by(["run", "step"])
        .agg(
            low=pl.col("magnitude").min(),
            high=pl.col("magnitude").max(),
        )
        .sort("step")
    )

    band = (
        alt.Chart(envelope)
        .mark_area(opacity=0.35)
        .encode(
            x=alt.X("step:Q", title="step"),
            y=alt.Y("low:Q", title="|value|", scale=alt.Scale(type="log")),
            y2="high:Q",
            color=alt.Color("run:N", title="run"),
            tooltip=["run:N", "step:Q", "low:Q", "high:Q"],
        )
        .properties(height=280, width=620)
    )
    mo.ui.altair_chart(band)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Latent weights against the quantization grid

    BitLinear only. `scaled_weight` is `weight * w_scale` -- the coordinates
    the quantizer sees. The rules at ±1 are the ternary clamp boundary:
    everything beyond them is already saturated, so mass drifting outside is
    capacity the forward pass cannot use.
    """)
    return


@app.cell
def _(hist, pl):
    scaled = (
        hist.filter((pl.col("kind") == "scaled_weight") & (pl.col("bin") != 0))
        .with_columns(
            fraction=pl.col("count") / pl.col("count").sum().over(["run_id", "step", "param"]),
            run=pl.col("run_id").str.slice(0, 8),
        )
    )
    return (scaled,)


@app.cell
def _(alt, mo, scaled):
    mo.stop(scaled.is_empty(), mo.md("_No BitLinear layers in the selected runs._"))

    base = alt.Chart(scaled)
    grid = base.mark_rect().encode(
        x=alt.X("step:O", title="step", axis=alt.Axis(labelOverlap=True)),
        y=alt.Y("lo:Q", title="weight x w_scale", scale=alt.Scale(type="symlog", constant=0.1)),
        y2="hi:Q",
        color=alt.Color("fraction:Q", title="share", scale=alt.Scale(type="log", scheme="magma")),
    )
    # Drawn with alt.datum off the same data as the heatmap: faceting a layered
    # chart requires every layer to share one top-level data source.
    boundaries = alt.layer(
        *(
            base.mark_rule(color="cyan", strokeDash=[4, 3]).encode(y=alt.datum(level))
            for level in (-1.0, 1.0)
        )
    )
    mo.ui.altair_chart(
        alt.layer(grid, boundaries)
        .properties(height=260, width=520)
        .facet(row=alt.Row("param:N", title=None))
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Saturation and occupancy

    Share of latent weights past the ternary boundary, and the share the
    quantizer maps to a nonzero level. These come from the histogram rather
    than the scalar metrics on purpose: under the default `AbsMedian` measure
    the scalar `frac_saturated` is pinned at 0.5 by construction, whereas the
    binned version can be read per magnitude.
    """)
    return


@app.cell
def _(alt, mo, pl, scaled):
    saturation = (
        scaled.with_columns(
            magnitude=pl.max_horizontal(pl.col("lo").abs(), pl.col("hi").abs())
        )
        .group_by(["run", "param", "step"])
        .agg(saturated=pl.col("fraction").filter(pl.col("magnitude") > 1.0).sum())
        .sort("step")
    )

    lines = (
        alt.Chart(saturation)
        .mark_line(point=False)
        .encode(
            x=alt.X("step:Q", title="step"),
            y=alt.Y("saturated:Q", title="share beyond ±1", scale=alt.Scale(zero=True)),
            color=alt.Color("param:N", title="layer"),
            strokeDash=alt.StrokeDash("run:N", title="run"),
            tooltip=["run:N", "param:N", "step:Q", "saturated:Q"],
        )
        .properties(height=260, width=620)
    )
    mo.ui.altair_chart(lines)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Scalar metrics

    Straight from the tracking server -- these are not recomputed from the
    histogram, and comparing them against it is a genuine cross-check.
    """)
    return


@app.cell
def _(mo):
    stat_picker = mo.ui.dropdown(
        options=["l2", "absmax", "abs_p99", "abs_p50", "frac_zero", "std"],
        value="frac_zero",
        label="Statistic",
    )
    metric_kind = mo.ui.dropdown(options=["grad", "weight"], value="grad", label="Kind")
    mo.hstack([metric_kind, stat_picker], justify="start", gap=2)
    return metric_kind, stat_picker


@app.cell
def _(alt, client, metric_kind, mo, pl, run_picker, stat_picker):
    rows = []
    for run_id in run_picker.value:
        run = client.get_run(run_id)
        prefix = f"{metric_kind.value}/{stat_picker.value}/"
        for key in run.data.metrics:
            if not key.startswith(prefix):
                continue
            for point in client.get_metric_history(run_id, key):
                rows.append(
                    {
                        "run": run_id[:8],
                        "param": key[len(prefix):],
                        "step": point.step,
                        "value": point.value,
                    }
                )
    mo.stop(not rows, mo.md("_No matching metrics._"))

    series = (
        alt.Chart(pl.DataFrame(rows))
        .mark_line()
        .encode(
            x=alt.X("step:Q", title="step"),
            y=alt.Y("value:Q", title=f"{metric_kind.value}/{stat_picker.value}"),
            color=alt.Color("param:N", title="parameter"),
            strokeDash=alt.StrokeDash("run:N", title="run"),
            tooltip=["run:N", "param:N", "step:Q", "value:Q"],
        )
        .properties(height=300, width=620)
    )
    mo.ui.altair_chart(series)
    return


if __name__ == "__main__":
    app.run()
