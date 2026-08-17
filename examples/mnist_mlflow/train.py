from __future__ import print_function
import argparse
from bitlinear import bitlinearize
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from watcher import ParamWatcher

import schedulefree


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, watcher=None):
    model.train()
    optimizer.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        step = (epoch - 1) * len(train_loader) + batch_idx
        if watcher is not None:
            # step() leaves .grad populated and zero_grad() is a whole iteration
            # away, so this sees the gradients that produced the current weights.
            watcher.sample(step)
        if batch_idx % args.log_interval == 0:
            mlflow.log_metric("train_loss", loss.item(), step=step)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, optimizer, device, test_loader):
    model.eval()
    optimizer.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--no-bitlinear', action='store_true', default=False,
                        help='disables BitLinear')
    parser.add_argument('--mlflow-uri', type=str, default='http://localhost:5001',
                        help='MLflow tracking server URI (default: http://localhost:5001)')
    parser.add_argument('--experiment', type=str, default='mnist-bitlinear',
                        help='MLflow experiment name (default: mnist-bitlinear)')
    parser.add_argument('--watch', action='store_true', default=False,
                        help='log per-layer weight and gradient statistics')
    parser.add_argument('--watch-interval', type=int, default=None, metavar='N',
                        help='batches between watcher samples (default: --log-interval)')
    parser.add_argument('--watch-eval-weights', action='store_true', default=False,
                        help='watch the schedule-free averaged weights instead of the training iterate')
    args = parser.parse_args()
    if args.watch_interval is None:
        args.watch_interval = args.log_interval
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    if not args.no_bitlinear:
        print("Replacing linear layers by BitLinear")
        bitlinearize(model, replacements=[{"strategy": "sample"}])
    print(model)
    model = model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    watcher = ParamWatcher(
        model,
        interval=args.watch_interval,
        optimizer=optimizer,
        eval_weights=args.watch_eval_weights,
    ) if args.watch else None

    with mlflow.start_run():
        params = {
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "bitlinear": not args.no_bitlinear,
        }
        if watcher is not None:
            params["watch_interval"] = args.watch_interval
            params["watch_weight_point"] = watcher.weight_point
        mlflow.log_params(params)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, watcher)
            test_loss, accuracy = test(model, optimizer, device, test_loader)
            mlflow.log_metrics({"test_loss": test_loss, "accuracy": accuracy}, step=epoch)

        if args.save_model:
            mlflow.pytorch.log_model(model, "model")
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
