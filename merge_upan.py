from argparse import ArgumentParser
import numpy as np
import torch
from utils import save_results
import mnist
import mnist_cifar10
import fmnist_kmnist
from mnist.dataloaders import mnist_combined_test_loader
from mnist_cifar10.dataloaders import (
    dual_channel_cifar10_test_loader,
    dual_channel_mnist_test_loader,
)
from fmnist_kmnist.dataloaders import fmnist_kmnist_test_loader
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN
from config import SEEDS, Config


def main(args):

    # Initialize arguments based on dataset chosen
    if args.dataset == "disjoint_mnist":
        args.arch = ["resnet18", "resnet18"]
    elif args.dataset == "mnist_cifar10":
        args.arch = ["resnet18", "resnet18"]

    # Initialize arguments based on testset chosen
    if args.testset == "disjoint_mnist":
        test_loader = mnist_combined_test_loader(args.test_batch_size)
        args.d1 = "first5_mnist"
        args.d2 = "last5_mnist"
        args.m1_input_channel = 1
        args.m2_input_channel = 1
        args.output_size = 5
        args.arch1 = "resnet18"
        args.arch2 = "resnet18"
        arch1 = ResNet18
        arch2 = ResNet18
        m = mnist
    elif args.testset == "mnist_cifar10":
        test_loader = [
            dual_channel_mnist_test_loader(args.test_batch_size),
            dual_channel_cifar10_test_loader(args.test_batch_size),
        ]
        args.d1 = "mnist"
        args.d2 = "cifar10"
        args.m1_input_channel = 1
        args.m2_input_channel = 3
        args.output_size = 10
        args.arch1 = "resnet18"
        args.arch2 = "resnet18"
        arch1 = ResNet18
        arch2 = ResNet18
        m = mnist_cifar10
    elif args.testset == "fmnist_kmnist":
        test_loader = fmnist_kmnist_test_loader(args.test_batch_size)
        args.d1 = "fmnist"
        args.d2 = "kmnist"
        args.m1_input_channel = 1
        args.m2_input_channel = 1
        args.output_size = 10
        args.arch1 = "resnet18"
        args.arch2 = "resnet18"
        arch1 = ResNet18
        arch2 = ResNet18
        m = fmnist_kmnist

    # Initialize logits statistics function
    experiment = m.smart_coordinator

    # Pan settings
    if args.upan_type == "logits":
        upan_input_size = args.output_size
        upan_arch = PAN
    elif args.upan_type == "agnostic_logits":
        upan_input_size = 3
        upan_arch = AgnosticPAN

    # Running the test
    print(f"Testset: {args.testset}")
    results = []

    for i in range(len(args.seeds)):
        seed = args.seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\nIteration: {i+1}, Seed: {seed}")

        # Load models
        expert1 = arch1(
            input_channel=args.m1_input_channel, output_size=args.output_size
        ).to(args.device)
        expert1.load_state_dict(
            torch.load(
                args.output_dir + f"{args.d1}_{args.arch1}_{args.seeds[i]}",
                map_location=torch.device(args.device),
            )
        )
        expert2 = arch2(
            input_channel=args.m2_input_channel, output_size=args.output_size
        ).to(args.device)
        expert2.load_state_dict(
            torch.load(
                args.output_dir + f"{args.d2}_{args.arch2}_{args.seeds[i]}",
                map_location=torch.device(args.device),
            )
        )

        # Running the experiment
        upan = upan_arch(input_size=upan_input_size).to(args.device)
        upan.load_state_dict(
            torch.load(
                args.upan_dir
                + f"upan_{args.upan_type}_{args.dataset}{args.arch}_{args.seeds[i]}",
                map_location=torch.device(args.device),
            )
        )
        result = experiment(args, expert1, expert2, upan, device, test_loader)

        # Adding more info to the result to be saved
        for r in result:
            r.update({"iteration": i, "seed": args.seeds[i]})
        results.extend(result)

    # Save the results
    if args.save_results:
        save_results(
            f"upan_{args.upan_type}_{args.dataset}_{args.testset}{[args.arch1, args.arch2]}",
            results,
            f"{args.results_dir}smart_coord(upan)/",
        )

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10"],
    )
    parser.add_argument(
        "--testset",
        type=str,
        default="disjoint_mnist",
        choices=["disjoint_mnist", "mnist_cifar10", "fmnist_kmnist"],
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="lenet5",
        choices=["lenet5", "lenet5_halfed", "resnet18"],
    )
    parser.add_argument(
        "--upan_type",
        type=str,
        default="agnostic_logits",
        choices=["logits", "agnostic_logits"],
    )
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_results", type=bool, default=True)
    parser.add_argument("--results_dir", type=str, default="./results/merge/")
    parser.add_argument("--output_dir", type=str, default="./cache/models/")
    parser.add_argument("--upan_dir", type=str, default="./cache/models/upan/")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.seeds = SEEDS
    args.device = device
    main(args)
