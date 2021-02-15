from argparse import ArgumentParser
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from utils import create_op_dir, save_results
from config import SEEDS, Config
from mnist.smart_coord import craft_disjoint_mnist_target
from mnist.dataloaders import mnist_combined_train_loader, mnist_combined_test_loader
from mnist_cifar10.smart_coord import craft_mnist_cifar10_target
from mnist_cifar10.dataloaders import mnist_cifar10_single_channel_train_loader, mnist_cifar10_single_channel_test_loader
from fmnist_kmnist.smart_coord import craft_fmnist_kmnist_target
from fmnist_kmnist.dataloaders import fmnist_kmnist_train_loader, fmnist_kmnist_test_loader
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN, compute_agnostic_stats
import time


def modify_target(targets, fpan_output_size, wrong_target_pct):
    temp = targets.clone().detach() # for debugging

    if wrong_target_pct:
        # Number of samples to be modified
        num_wrong_target_samples = int(wrong_target_pct * len(targets))
        # Indices of samples to be modified
        indices = random.sample(range(len(targets)), num_wrong_target_samples)
        for index in indices:
            # Change the target of the selected sample to a random incorrect target
            targets[index] = random.choice(
                list(range(targets[index])) + list(range(targets[index]+1, fpan_output_size)))

    print("original:", len(temp)) # for debugging
    print("correct:", temp.eq(targets).sum().item())
    return targets


def create_data_loader(device, data_loader, target_create_fn, batch_size, fpan_output_size, wrong_target_pct=0):
    fpan_input = []
    fpan_target = []
    # Using input data of the test loader as input data of FPAN, and generate target for testing
    for data, target in data_loader:
        fpan_input.append(data.to(device))
        fpan_target.append(target_create_fn(target.to(device)).to(device))
    fpan_input = torch.cat(fpan_input)
    fpan_target = torch.cat(fpan_target)

    # Modify targets
    fpan_target = modify_target(fpan_target, fpan_output_size, wrong_target_pct)

    # Slice dataset into smallest unit
    fpan_data = []
    for idx in range(len(fpan_input)):
        fpan_data.append([fpan_input[idx], fpan_target[idx]])

    return DataLoader(fpan_data, batch_size=batch_size, shuffle=False)


def train(args, fpan, train_loader, optimizer, epoch):
    fpan.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = fpan(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train FPAN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx * len(data) / len(train_loader.dataset),
                    loss.item(),
                )
            )


def test(args, fpan, test_loader):
    fpan.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = fpan(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print("\nTest set: Accuracy: {}/{} ({:.0f}%)".format(correct, len(test_loader.dataset), acc))
    return test_loss, acc


def train_model(fpan, trial, device, train_loader, test_loader, target_create_fn, fpan_output_size, config_args):
    fpan = fpan.to(device)
    optimizer = optim.SGD(fpan.parameters(), lr=config_args.lr, momentum=config_args.momentum)

    # Initialise dataloaders of FPAN
    fpan_train_loader = create_data_loader(device, train_loader, target_create_fn, config_args.batch_size, fpan_output_size, config_args.wrong_target_pct,)
    fpan_test_loader = create_data_loader(device, test_loader, target_create_fn, config_args.test_batch_size, fpan_output_size)

    for epoch in range(1, config_args.epochs + 1):
        start_time = time.time()
        # Train fpan
        train(config_args, fpan, fpan_train_loader, optimizer, epoch)
        # Test fpan
        test_loss, acc = test(config_args, fpan, fpan_test_loader)
        print('Time taken: {}.\n'.format(time.time() - start_time))

    return fpan, test_loss, acc


def train_fpan(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on the chosen dataset
    if args.fpan_data == "disjoint_mnist":
        train_loaders = mnist_combined_train_loader(args.batch_size)
        test_loaders = mnist_combined_test_loader(args.test_batch_size)
        target_create_fns = craft_disjoint_mnist_target
        fpan_output_size = 2
        fpan_input_channel = 1
    elif args.fpan_data == "mnist_cifar10":
        train_loaders = mnist_cifar10_single_channel_train_loader(args.batch_size)
        test_loaders = mnist_cifar10_single_channel_test_loader(args.test_batch_size)
        target_create_fns = craft_mnist_cifar10_target
        fpan_output_size = 2
        fpan_input_channel = 1
    elif args.fpan_data == "fmnist_kmnist":
        train_loaders = fmnist_kmnist_train_loader(args.batch_size)
        test_loaders = fmnist_kmnist_test_loader(args.test_batch_size)
        target_create_fns = craft_fmnist_kmnist_target
        fpan_output_size = 2
        fpan_input_channel = 1

    if args.fpan_arch == "resnet18":
        fpan_arch = ResNet18
    elif args.fpan_arch == "lenet5":
        fpan_arch = LeNet5
    elif args.fpan_arch == "lenet5_halfed":
        fpan_arch = LeNet5Halfed

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"\nFPAN Dataset: {args.fpan_data}")
    print(f"FPAN arch: {args.fpan_arch}")
    print(f"Wrong Targets Percentage: {args.wrong_target_pct}\n")

    fpan_results = []
    for i in range(len(args.seeds)):
        print(f"Iteration {i+1}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])

        # Train FPAN model
        fpan, fpan_test_loss, fpan_acc = train_model(
            fpan=fpan_arch(input_channel=fpan_input_channel, output_size=fpan_output_size).to(device),
            trial=i,
            device=device,
            train_loader=train_loaders,
            test_loader=test_loaders,
            target_create_fn=target_create_fns,
            fpan_output_size=fpan_output_size,
            config_args=args,
        )

        # Save the FPAN model
        if args.save_models:
            torch.save(
                fpan.state_dict(),
                args.output_dir
                + f"fpan_{args.fpan_data}({fpan_input_channel})_(wrong pct={args.wrong_target_pct}){args.seeds[i]}",
            )

        # Save the results in list first
        fpan_results.append(
            {
                "iteration": i,
                "seed": args.seeds[i],
                "loss": fpan_test_loss,
                "acc": fpan_acc,
            }
        )

    # Save all the results
    if args.save_results:
        save_results(
            f"fpan_{args.fpan_data}({fpan_input_channel})_(wrong pct={args.wrong_target_pct})",
            fpan_results,
            args.results_dir,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--fpan_data",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--fpan_arch",
        type=str,
        default="resnet18",
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
    parser.add_argument("--results_dir", type=str, default="./results/experiments/fpan_trained_with_wrong_targets/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/experiments/fpan_trained_with_wrong_targets/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_fpan(args)
