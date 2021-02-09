from argparse import ArgumentParser
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from utils import create_op_dir, save_results
from config import SEEDS, Config
from mnist.datasets import (
    mnist_first5_train_dataset,
    mnist_first5_test_dataset,
    mnist_last5_train_dataset,
    mnist_last5_test_dataset,
)
from mnist_cifar10.datasets import (
    mnist_train_dataset,
    mnist_test_dataset,
    cifar10_train_dataset,
    cifar10_test_dataset,
)
from fmnist_kmnist.datasets import (
    fmnist_train_dataset,
    fmnist_test_dataset,
    kmnist_train_dataset,
    kmnist_test_dataset,
)
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18


def modify_dataset(dataset, wrong_target_pct):
    if wrong_target_pct:
        # Number of samples to be modified
        num_wrong_target_samples = int(wrong_target_pct * len(dataset.targets))
        # Indices of samples to be modified
        indices = random.sample(range(len(dataset.targets)), num_wrong_target_samples)
        for index in indices:
            # Change the target of the selected sample to a random incorrect target
            dataset.targets[index] = random.choice(
                list(range(dataset.targets[index])) + list(range(dataset.targets[index]+1, len(dataset.classes))))
    return dataset


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), acc,
        )
    )
    return test_loss, acc


def train_model(model, device, train_loader, test_loader, config_args):
    model = model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config_args.lr, momentum=config_args.momentum
    )

    for epoch in range(1, config_args.epochs + 1):
        train(config_args, model, device, train_loader, optimizer, epoch)
        test_loss, acc = test(config_args, model, device, test_loader)
    return model, test_loss, acc


def train_main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on dataset chosen
    if args.dataset == "first5_mnist":
        train_dataset = mnist_first5_train_dataset
        test_dataset = mnist_first5_test_dataset
        args.output_size = 5
        args.input_channel = 1
    elif args.dataset == "last5_mnist":
        train_dataset = mnist_last5_train_dataset
        test_dataset = mnist_last5_test_dataset
        args.output_size = 5
        args.input_channel = 1
    elif args.dataset == "mnist":
        train_dataset = mnist_train_dataset
        test_dataset = mnist_test_dataset
        args.output_size = 10
        args.input_channel = 1
    elif args.dataset == "cifar10":
        train_dataset = cifar10_train_dataset
        test_dataset = cifar10_test_dataset
        args.output_size = 10
        args.input_channel = 3
    elif args.dataset == "fmnist":
        train_dataset = fmnist_train_dataset
        test_dataset = fmnist_test_dataset
        args.output_size = 10
        args.input_channel = 1
    elif args.dataset == "kmnist":
        train_dataset = kmnist_train_dataset
        test_dataset = kmnist_test_dataset
        args.output_size = 10
        args.input_channel = 1

    # Modify targets of dataset
    modified_train_dataset = modify_dataset(train_dataset, args.wrong_target_pct)

    # Create data loader
    train_loader = DataLoader(modified_train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Initialize models based on architecture chosen
    if args.arch == "lenet5":
        arch = LeNet5
    elif args.arch == "lenet5_halfed":
        arch = LeNet5Halfed
    elif args.arch == "resnet18":
        arch = ResNet18

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"\nDataset: {args.dataset}")
    print(f"Model: {args.arch}")
    print(f"Wrong Targets Percentage: {args.wrong_target_pct}\n")
    results = []

    for i in range(len(args.seeds)):
        print(f"Iteration {i+1}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])

        model, test_loss, acc = train_model(
            arch(input_channel=args.input_channel, output_size=args.output_size),
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            config_args=args,
        )

        # Save the model
        if args.save_models:
            torch.save(
                model.state_dict(),
                args.output_dir + f"{args.dataset}(wrong pct={args.wrong_target_pct})_{args.arch}_{args.seeds[i]}",
            )

        # Save the results in list first
        results.append(
            {"iteration": i, "seed": args.seeds[i], "loss": test_loss, "acc": acc}
        )

    # Save all the results
    if args.save_results:
        save_results(f"{args.dataset}(wrong pct={args.wrong_target_pct})_{args.arch}", results, args.results_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["first5_mnist", "last5_mnist", "mnist", "cifar10", "fmnist", "kmnist"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="lenet5",
        choices=["lenet5", "lenet5_halfed", "resnet18"],
    )
    parser.add_argument("--wrong_target_pct", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr, help="learning rate")
    parser.add_argument("--momentum", type=float, default=Config.momentum)
    parser.add_argument("--no_cuda", type=bool, default=Config.no_cuda)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--save_results", type=bool, default=Config.save_results)
    parser.add_argument("--save_models", type=bool, default=False)
    parser.add_argument("--results_dir", type=str, default="./results/experiments/source_net_trained_with_wrong_targets/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/experiments/source_net_trained_with_wrong_targets/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_main(args)
