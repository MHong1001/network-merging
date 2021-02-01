from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from utils import create_op_dir, save_results
from config import SEEDS, Config
from mnist.smart_coord import craft_disjoint_mnist_target
from mnist.dataloaders import mnist_combined_train_loader_noshuffle, mnist_combined_test_loader
from mnist_cifar10.smart_coord import craft_mnist_cifar10_target
from mnist_cifar10.dataloaders import (
    mnist_cifar10_single_channel_train_loader_noshuffle,
    mnist_cifar10_single_channel_test_loader,
    mnist_cifar10_3_channel_train_loader_noshuffle,
    mnist_cifar10_3_channel_test_loader,
)
from fmnist_kmnist.smart_coord import craft_fmnist_kmnist_target
from fmnist_kmnist.dataloaders import fmnist_kmnist_train_loader_noshuffle, fmnist_kmnist_test_loader
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN, compute_agnostic_stats
import time


def eval_expert(args, expert_idx, expert, device, data_loader):
    expert.eval()
    expert_output = []  # collect output logits of expert
    total_data = sum(len(data) for data, target in data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        logits = expert(data, out_feature=False)    # output logits of expert
        expert_output.append(logits.detach())
        del logits

        if batch_idx % args.log_interval == 0:
            print(
                "Eval Expert: {} [{}/{} ({:.0f}%)]".format(
                    expert_idx + 1,
                    batch_idx * len(data),
                    total_data,
                    100.0 * batch_idx * len(data) / total_data,
                )
            )
    return expert_output


def eval_upan(args, upan, all_experts_output, device):
    upan.eval()
    fpan_target = []
    for expert_output in all_experts_output:
        upan_output = []
        for logits in expert_output:
            if args.upan_type == "logits":
                output = upan(logits)
            elif args.upan_type == "agnostic_logits":
                output = upan(compute_agnostic_stats(logits))
            output = F.log_softmax(output, dim=-1)
            upan_output.append(output)

        # Concatenate batches of UPAN outputs
        upan_output = torch.cat(upan_output)
        # Extract the output of UPAN (ie. probability of the expert truly belonging to the input data)
        upan_output = torch.index_select(upan_output, 1, torch.tensor([1]).to(device))
        upan_output = torch.flatten(upan_output)
        fpan_target.append(upan_output)

    # Concatenate UPAN predictions on different experts when given the same input data
    fpan_target = torch.stack(fpan_target, dim=1)
    # Extract index of the max log-probability (represents the expert chosen by UPAN)
    fpan_target = torch.argmax(fpan_target, dim=1)
    return fpan_target


def create_train_loader(args, device, all_experts, upan, train_loader):
    print("Obtaining FPAN training set:")
    # Using input data of the first dataloader as input data of FPAN
    fpan_input = []
    for data, target in train_loader[0]:
        fpan_input.append(data.to(device))
    fpan_input = torch.cat(fpan_input)

    # Feed data to each expert to collect their output logits
    all_experts_output = []
    for expert_idx, expert in enumerate(all_experts):
        expert_output = eval_expert(
            args,
            expert_idx,
            expert,
            device,
            train_loader[expert_idx],
        )
        all_experts_output.append(expert_output)

    # UPAN selects an expert as the target of FPAN
    fpan_target = eval_upan(args, upan, all_experts_output, device)

    # Slice dataset into the smallest unit
    fpan_train_data = []
    for idx in range(len(fpan_input)):
        fpan_train_data.append((fpan_input[idx], fpan_target[idx]))

    return DataLoader(fpan_train_data, batch_size=args.batch_size, shuffle=True)


def create_test_loader(args, device, test_loader, target_create_fn):
    print("Obtaining FPAN test set:")
    fpan_input = []
    fpan_target = []
    # Using input data of the test loader as input data of FPAN, and generate target for testing
    for data, target in test_loader:
        fpan_input.append(data.to(device))
        fpan_target.append(target_create_fn(target.to(device)).to(device))
    fpan_input = torch.cat(fpan_input)
    fpan_target = torch.cat(fpan_target)

    # Slice dataset into smallest unit
    fpan_test_data = []
    for idx in range(len(fpan_input)):
        fpan_test_data.append([fpan_input[idx], fpan_target[idx]])

    return DataLoader(fpan_test_data, batch_size=args.test_batch_size, shuffle=False)


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


def train_model(fpan, trial, device, expert_arch, train_loader, test_loader, target_create_fn, config_args):
    fpan = fpan.to(device)
    optimizer = optim.SGD(fpan.parameters(), lr=config_args.lr, momentum=config_args.momentum)

    # Load experts
    all_experts = []
    for expert_idx in range(len(expert_arch)):
        expert = expert_arch[expert_idx](
            input_channel=args.expert_input_channel[expert_idx], output_size=args.expert_output_size
        ).to(device)
        expert.load_state_dict(
            torch.load(
                args.model_dir +
                f"{args.expert[expert_idx]}_{args.expert_arch[expert_idx]}_{args.seeds[trial]}",
                map_location=torch.device(device),
            )
        )
        all_experts.append(expert)

    # Load UPAN model
    if args.upan_type == "logits":
        upan_input_size = args.model_output_size
        upan_arch = PAN
    elif args.upan_type == "agnostic_logits":
        upan_input_size = 3
        upan_arch = AgnosticPAN

    upan = upan_arch(input_size=upan_input_size).to(device)
    upan.load_state_dict(
        torch.load(
            args.upan_dir
            + f"upan_{args.upan_type}_{args.upan_data}{args.model_arch}_{args.seeds[trial]}",
            map_location=torch.device(device),
        )
    )

    # Initialise dataloaders of FPAN
    fpan_train_loader = create_train_loader(args, device, all_experts, upan, train_loader)
    fpan_test_loader = create_test_loader(args, device, test_loader, target_create_fn)

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
        train_loaders = [
            mnist_combined_train_loader_noshuffle(args.batch_size),
            mnist_combined_train_loader_noshuffle(args.batch_size),
        ]
        test_loaders = mnist_combined_test_loader(args.test_batch_size)
        args.expert = ["first5_mnist", "last5_mnist"]
        args.expert_input_channel = [1, 1]
        args.expert_output_size = 5
        args.expert_arch = ["resnet18", "resnet18"]
        expert_arch = [ResNet18, ResNet18]
        target_create_fns = craft_disjoint_mnist_target
        fpan_output_size = 2
        fpan_input_channel = 1
    elif args.fpan_data == "mnist_cifar10":
        # DELETE THIS COMMENT WHEN SUBMITTING
        # CIFAR10 is converted into single channel, instead of converting MNIST into RGB,
        # otherwise FPAN can just easily distinguish by recognizing the colour of the image.
        train_loaders = [
            mnist_cifar10_single_channel_train_loader_noshuffle(args.batch_size),
            mnist_cifar10_3_channel_train_loader_noshuffle(args.batch_size),
        ]
        test_loaders = mnist_cifar10_single_channel_test_loader(args.test_batch_size)
        args.expert = ["mnist", "cifar10"]
        args.expert_input_channel = [1, 3]
        args.expert_output_size = 10
        args.expert_arch = ["resnet18", "resnet18"]
        expert_arch = [ResNet18, ResNet18]
        target_create_fns = craft_mnist_cifar10_target
        fpan_output_size = 2
        fpan_input_channel = 1
    elif args.fpan_data == "fmnist_kmnist":
        train_loaders = [
            fmnist_kmnist_train_loader_noshuffle(args.batch_size),
            fmnist_kmnist_train_loader_noshuffle(args.batch_size),
        ]
        test_loaders = fmnist_kmnist_test_loader(args.test_batch_size)
        args.expert = ["fmnist", "kmnist"]
        args.expert_input_channel = [1, 1]
        args.expert_output_size = 10
        args.expert_arch = ["resnet18", "resnet18"]
        expert_arch = [ResNet18, ResNet18]
        target_create_fns = craft_fmnist_kmnist_target
        fpan_output_size = 2
        fpan_input_channel = 1

    if args.fpan_arch == "resnet18":
        fpan_arch = ResNet18
    elif args.fpan_arch == "lenet5":
        fpan_arch = LeNet5
    elif args.fpan_arch == "lenet5_halfed":
        fpan_arch = LeNet5Halfed

    # Initialize arguments based on the dataset that UPAN was trained on
    if args.upan_data == "disjoint_mnist":
        args.model_arch = ["resnet18", "resnet18"]
        args.model_output_size = 5
    elif args.upan_data == "mnist_cifar10":
        args.model_arch = ["resnet18", "resnet18"]
        args.model_output_size = 10
    elif args.upan_data == "fmnist_kmnist":
        args.model_arch = ["resnet18", "resnet18"]
        args.model_output_size = 10

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"\nFPAN Dataset: {args.fpan_data}")
    print(f"FPAN arch: {args.fpan_arch}")
    print(f"UPAN Dataset: {args.upan_data}")
    print(f"UPAN type: {args.upan_type}\n")

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
            expert_arch=expert_arch,
            train_loader=train_loaders,
            test_loader=test_loaders,
            target_create_fn=target_create_fns,
            config_args=args,
        )

        # Save the FPAN model
        torch.save(
            fpan.state_dict(),
            args.output_dir
            + f"fpan_{args.fpan_data}({fpan_input_channel})_({args.upan_data}_{args.upan_type}){args.seeds[i]}",
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
            f"fpan_{args.fpan_data}({fpan_input_channel})_({args.upan_data}_{args.upan_type})",
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
    parser.add_argument(
        "--upan_data",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--upan_type",
        type=str,
        default="agnostic_logits",
        choices=["agnostic_logits", "feature", "logits"],
    )
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr, help="learning rate")
    parser.add_argument("--momentum", type=float, default=Config.momentum)
    parser.add_argument("--no_cuda", type=bool, default=Config.no_cuda)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--save_results", type=bool, default=Config.save_results)
    parser.add_argument("--results_dir", type=str, default="./results/fpan/")
    parser.add_argument("--model_dir", type=str, default="./cache/models/")
    parser.add_argument("--upan_dir", type=str, default="./cache/models/upan/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/fpan/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_fpan(args)
