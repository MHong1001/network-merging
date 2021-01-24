from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from utils import create_op_dir, save_results
from config import SEEDS, Config
from mnist.smart_coord import craft_first_5_target, craft_last_5_target
from mnist.dataloaders import mnist_combined_train_loader, mnist_combined_test_loader
from mnist_cifar10.smart_coord import craft_mnist_target, craft_cifar10_target
from mnist_cifar10.dataloaders import (
    mnist_cifar10_single_channel_train_loader,
    mnist_cifar10_single_channel_test_loader,
    mnist_cifar10_3_channel_train_loader,
    mnist_cifar10_3_channel_test_loader,
)
from fmnist_kmnist.smart_coord import craft_fmnist_target, craft_kmnist_target
from fmnist_kmnist.dataloaders import fmnist_kmnist_train_loader, fmnist_kmnist_test_loader
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN, compute_agnostic_stats
import time


def eval_expert(args, expert_idx, expert, device, data_loader, target_create_fn):
    expert.eval()
    upan_dataset = []    # collect output of expert and target for UPAN
    total_data = sum(len(data) for data, target in data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        logits = expert(data, out_feature=False)    # output logits of expert
        upan_target = target_create_fn(target).to(device)   # target for UPAN

        if args.upan_type == "logits" or args.upan_type == "agnostic_logits":
            upan_dataset.append((logits.detach(), upan_target))
        else:
            raise NotImplementedError("Not an eligible upan type.")
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
    del expert, data_loader
    return upan_dataset


def create_train_loader(args, trial, train_arch, device, train_loader, target_create_fn, config_args):
    # Feed data to each expert, return training set of upan
    print("Obtaining UPAN training set:")
    all_train_data = []
    for expert_idx in range(len(train_arch)):
        # Load expert
        expert = train_arch[expert_idx](
            input_channel=args.train_input_channel[expert_idx], output_size=args.train_output_size
        ).to(device)
        expert.load_state_dict(
            torch.load(
                args.expert_dir +
                f"{args.train_expert[expert_idx]}_{args.train_arch[expert_idx]}_{args.seeds[trial]}",
                map_location=torch.device(device),
            )
        )
        train_data = eval_expert(
            config_args,
            expert_idx,
            expert,
            device,
            train_loader[expert_idx],
            target_create_fn[expert_idx]
        )

        # Slice dataset into the smallest unit
        temp_train_data = []
        for output, upan_target in train_data:
            for idx in range(len(output)):
                temp_train_data.append((output[idx], upan_target[idx]))
        all_train_data.append(temp_train_data)
        del expert, train_data, temp_train_data
    return DataLoader(ConcatDataset(all_train_data), batch_size=args.batch_size, shuffle=True)


def create_test_loader(args, trial, test_arch, device, test_loader, test_target_create_fn, config_args):
    # Feed test data to expert, return test set of upan
    print("Obtaining UPAN test set:")
    upan_test_loader = []
    for expert_idx in range(len(test_arch)):
        # Load expert
        expert = test_arch[expert_idx](
            input_channel=args.test_input_channel[expert_idx], output_size=args.test_output_size
        ).to(device)
        expert.load_state_dict(
            torch.load(
                args.expert_dir +
                f"{args.test_expert[expert_idx]}_{args.test_arch[expert_idx]}_{args.seeds[trial]}",
                map_location=torch.device(device),
            )
        )

        # Directly append output of expert (not slicing into smallest unit)
        upan_test_loader.append(
            eval_expert(
                config_args,
                expert_idx,
                expert,
                device,
                test_loader[expert_idx],
                test_target_create_fn[expert_idx]
            )
        )
        del expert
    return upan_test_loader


def train(args, upan, upan_train_loader, optimizer, epoch):
    # Use the collected training set to train upan
    upan.train()
    total_data = sum(len(data) for data, target in upan_train_loader)
    for batch_idx, (data, upan_target) in enumerate(upan_train_loader):
        if args.upan_type == "feature" or args.upan_type == "logits":
            output = upan(data)
        elif args.upan_type == "agnostic_feature" or args.upan_type == "agnostic_logits":
            output = upan(compute_agnostic_stats(data))

        optimizer.zero_grad()
        loss = F.cross_entropy(output, upan_target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train UPAN Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    total_data,
                    100.0 * batch_idx * len(data) / total_data,
                    loss.item(),
                )
            )


def test(args, device, upan, upan_test_loader):
    upan.eval()
    test_loss = "N/A"   # Not being used
    with torch.no_grad():
        model_pred = []
        model_target = []
        for expert_data in upan_test_loader:
            upan_output = []
            upan_target = []
            for logits, target in expert_data:
                if args.upan_type == "logits":
                    output = upan(logits)
                elif args.upan_type == "agnostic_logits":
                    output = upan(compute_agnostic_stats(logits))
                output = F.log_softmax(output, dim=-1)
                upan_output.append(output)
                upan_target.append(target)

            # Concatenate batches of UPAN outputs and targets
            upan_output = torch.cat(upan_output)
            upan_target = torch.cat(upan_target)

            # Extract the output of UPAN (ie. probability of the expert truly belonging to the input data)
            upan_output = torch.index_select(upan_output, 1, torch.tensor(1).to(device))
            upan_output = torch.flatten(upan_output)

            # Append UPAN output and target for this expert
            model_pred.append(upan_output)
            model_target.append(upan_target)

        # Concatenate UPAN predictions on different experts when given the same input data
        model_pred = torch.stack(model_pred, dim=1)
        # Extract index of the max log-probability (represents the expert chosen by UPAN)
        model_pred = torch.argmax(model_pred, dim=1)

        # Concatenate UPAN targets on different experts when given the same input data
        model_target = torch.stack(model_target, dim=1)
        # Extract index of the true target (represent the correct expert)
        model_target = torch.nonzero(model_target, as_tuple=True)[1]

        correct = model_pred.eq(model_target).sum().item()
        total_data = len(model_pred)

    acc = 100.0 * correct / total_data
    print("\nTest set: Accuracy: {}/{} ({:.0f}%)".format(correct, total_data, acc))
    return test_loss, acc


def train_model(
    upan, trial, train_arch, test_arch, device, train_loader, test_loader, target_create_fn, test_target_create_fn, config_args
):
    optimizer = optim.SGD(upan.parameters(), lr=config_args.lr, momentum=config_args.momentum)

    # Initialise dataloaders of UPAN
    upan_train_loader = create_train_loader(args, trial, train_arch, device, train_loader, target_create_fn, config_args)
    upan_test_loader = create_test_loader(args, trial, test_arch, device, test_loader, test_target_create_fn, config_args)

    for epoch in range(1, config_args.epochs + 1):
        start_time = time.time()
        # Train upan
        train(args, upan, upan_train_loader, optimizer, epoch)
        # Test upan
        test_loss, acc = test(config_args, device, upan, upan_test_loader)
        print('Time taken: {}.\n'.format(time.time() - start_time))

    return upan, test_loss, acc


def train_upan(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize arguments based on the chosen dataset
    if args.dataset == "disjoint_mnist":
        train_loaders = [
            mnist_combined_train_loader(args.batch_size),
            mnist_combined_train_loader(args.batch_size),
        ]
        args.train_expert = ["first5_mnist", "last5_mnist"]
        args.train_input_channel = [1, 1]
        args.train_output_size = 5
        args.train_arch = ["resnet18", "resnet18"]
        train_arch = [ResNet18, ResNet18]
        target_create_fns = [craft_first_5_target, craft_last_5_target]
    elif args.dataset == "mnist_cifar10":
        train_loaders = [
            mnist_cifar10_single_channel_train_loader(args.batch_size),
            mnist_cifar10_3_channel_train_loader(args.batch_size),
        ]
        args.train_expert = ["mnist", "cifar10"]
        args.train_input_channel = [1, 3]
        args.train_output_size = 10
        args.train_arch = ["resnet18", "resnet18"]
        train_arch = [ResNet18, ResNet18]
        target_create_fns = [craft_mnist_target, craft_cifar10_target]
    elif args.dataset == "fmnist_kmnist":
        train_loaders = [
            fmnist_kmnist_train_loader(args.batch_size),
            fmnist_kmnist_train_loader(args.batch_size),
        ]
        args.train_expert = ["fmnist", "kmnist"]
        args.train_input_channel = [1, 1]
        args.train_output_size = 10
        args.train_arch = ["resnet18", "resnet18"]
        train_arch = [ResNet18, ResNet18]
        target_create_fns = [craft_fmnist_target, craft_kmnist_target]

    # Initialize arguments based on the chosen testset 
    if args.testset == "disjoint_mnist":
        test_loaders = [
            mnist_combined_test_loader(args.test_batch_size),
            mnist_combined_test_loader(args.test_batch_size),
        ]
        args.test_expert = ["first5_mnist", "last5_mnist"]
        args.test_input_channel = [1, 1]
        args.test_output_size = 5
        args.test_arch = ["resnet18", "resnet18"]
        test_arch = [ResNet18, ResNet18]
        test_target_create_fns = [craft_first_5_target, craft_last_5_target]
    elif args.testset == "mnist_cifar10":
        test_loaders = [
            mnist_cifar10_single_channel_test_loader(args.test_batch_size),
            mnist_cifar10_3_channel_test_loader(args.test_batch_size),
        ]
        args.test_expert = ["mnist", "cifar10"]
        args.test_input_channel = [1, 3]
        args.test_output_size = 10
        args.test_arch = ["resnet18", "resnet18"]
        test_arch = [ResNet18, ResNet18]
        test_target_create_fns = [craft_mnist_target, craft_cifar10_target]
    elif args.testset == "fmnist_kmnist":
        test_loaders = [
            fmnist_kmnist_test_loader(args.test_batch_size),
            fmnist_kmnist_test_loader(args.test_batch_size),
        ]
        args.test_expert = ["fmnist", "kmnist"]
        args.test_input_channel = [1, 1]
        args.test_output_size = 10
        args.test_arch = ["resnet18", "resnet18"]
        test_arch = [ResNet18, ResNet18]
        test_target_create_fns = [craft_fmnist_target, craft_kmnist_target]

    # Initialize UPAN based on its type
    if args.upan_type == "logits":
        upan_input_size = args.train_output_size # output size of expert
        upan_arch = PAN
    elif args.upan_type == "agnostic_logits":
        upan_input_size = 4 # number of statistical functions used
        upan_arch = AgnosticPAN

    # Create the directory for saving if it does not exist
    create_op_dir(args.output_dir)

    print(f"\nDataset: {args.dataset}")
    print(f"Testset: {args.testset}")
    print(f"UPAN type: {args.upan_type}\n")

    upan_results = []
    for i in range(len(args.seeds)):
        print(f"Iteration {i+1}, Seed {args.seeds[i]}")

        np.random.seed(args.seeds[i])
        torch.manual_seed(args.seeds[i])

        # Train UPAN model
        upan, upan_test_loss, upan_acc = train_model(
            upan=upan_arch(input_size=upan_input_size).to(device),
            trial=i,
            train_arch=train_arch,
            test_arch=test_arch,
            device=device,
            train_loader=train_loaders,
            test_loader=test_loaders,
            target_create_fn=target_create_fns,
            test_target_create_fn=test_target_create_fns,
            config_args=args,
        )

        # Save the UPAN model
        torch.save(
            upan.state_dict(),
            args.output_dir
            + f"upan_{args.upan_type}_{args.dataset}{args.train_arch}_{args.seeds[i]}",
        )

        # Save the results in list first
        upan_results.append(
            {
                "iteration": i,
                "seed": args.seeds[i],
                "loss": upan_test_loss,
                "acc": upan_acc,
            }
        )

    # Save all the results
    if args.save_results:
        save_results(
            f"upan_{args.upan_type}_{args.dataset}{args.train_arch}_{args.testset}{args.test_arch}",
            upan_results,
            args.results_dir,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--upan_type",
        type=str,
        default="agnostic_logits",
        choices=["agnostic_logits", "logits"],
    )
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr, help="learning rate")
    parser.add_argument("--momentum", type=float, default=Config.momentum)
    parser.add_argument("--no_cuda", type=bool, default=Config.no_cuda)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--save_results", type=bool, default=Config.save_results)
    parser.add_argument("--results_dir", type=str, default="./results/upan/")
    parser.add_argument("--expert_dir", type=str, default="./cache/models/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/upan/")

    args = parser.parse_args()
    args.seeds = SEEDS

    train_upan(args)
