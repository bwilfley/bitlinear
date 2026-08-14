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

## View results

Open **http://localhost:5001** in a browser. Runs appear under the
`mnist-bitlinear` experiment as they log.
