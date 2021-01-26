from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
from utils import save_results
import mnist
# import mnist_cifar10
# import fmnist_kmnist
from mnist.dataloaders import mnist_combined_test_loader
# from mnist_cifar10.dataloaders import (
#     dual_channel_cifar10_test_loader,
#     dual_channel_mnist_test_loader,
# )
# from fmnist_kmnist.dataloaders import fmnist_kmnist_test_loader
from archs.lenet5 import LeNet5, LeNet5Halfed
from archs.resnet import ResNet18
from archs.pan import PAN, AgnosticPAN, compute_agnostic_stats
from config import SEEDS, Config

import sys


def eval_expert(args, expert_idx, expert, data_loader):
    expert.eval()
    expert_logits = []    # collect output of expert and target for UPAN
    total_data = sum(len(data) for data, target in data_loader)
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        logits = expert(data, out_feature=False)    # output logits of expert

        if args.upan_type == "logits" or args.upan_type == "agnostic_logits":
            expert_logits.append((logits.detach()))
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
    return expert_logits


def smart_coordinator(args, upan, all_experts, test_loaders):
    # get target of smart coordinator
    sc_target = []
    for data, target in test_loaders[0]:
        target = target.to(args.device)
        sc_target.append(target)
    sc_target = torch.cat(sc_target)

    # feed data to experts
    all_experts_logits = []
    for expert_idx, expert in enumerate(all_experts):
        expert_logits = eval_expert(args, expert_idx, expert, test_loaders[expert_idx])
        all_experts_logits.append(expert_logits)

    # evaluate UPAN
    upan.eval()
    model_pred = []
    for expert_logits in all_experts_logits:
        upan_output = []
        for logits in expert_logits:
            if args.upan_type == "logits":
                output = upan(logits)
            elif args.upan_type == "agnostic_logits":
                output = upan(compute_agnostic_stats(logits))
            output = F.log_softmax(output, dim=-1)
            upan_output.append(output)

        # Concatenate batches of UPAN outputs
        upan_output = torch.cat(upan_output)

        # Extract the output of UPAN (ie. probability of the expert truly belonging to the input data)
        upan_output = torch.index_select(upan_output, 1, torch.tensor(1).to(device))
        upan_output = torch.flatten(upan_output)

        # Append UPAN output and target for this expert
        model_pred.append(upan_output)

    # Concatenate UPAN predictions on different experts when given the same input data
    model_pred = torch.stack(model_pred, dim=1)
    # Extract index of the max log-probability (represents the expert chosen by UPAN)
    model_pred = torch.argmax(model_pred, dim=1)


    """
    :all_combined_output = [[batch 1 of logits from expert 1, batch 2 of logtis from expert 1, ...], 
                            [batch 1 of logits from expert 2, batch 2 of logtis from expert 2, ...]]
    :model_pred = tensor([0, 1, 1, ..., 1])
    """
    all_combined_output = []
    for i in range(len(all_experts_logits[0])):
        output1 = all_experts_logits[0][i]
        output2 = all_experts_logits[1][i]
        for j in range(len(output1)):
            # j loops from 1 to 128
            #
            if model_pred[i*args.test_batch_size + j] == torch.tensor(0).to(args.device):
                # p1 true and p2 false
                combined_output = torch.cat(
                    [
                        output1[j],
                        torch.Tensor([torch.min(output1[j])] * len(output1[j])).to(
                            args.device
                        ),
                    ]
                )
            else:
                # p1 false and p2 true
                combined_output = torch.cat(
                    [
                        torch.Tensor([torch.min(output2[j])] * len(output2[j])).to(
                            args.device
                        ),
                        output2[j],
                    ]
                )
            all_combined_output.append(combined_output)
    all_combined_output = torch.stack(all_combined_output, dim=0)

    test_loss = F.cross_entropy(
        all_combined_output, sc_target, reduction="sum"
    ).item()  # sum up batch loss
    pred = all_combined_output.argmax(
        dim=1, keepdim=False
    )  # get the index of the max log-probability
    correct = pred.eq(sc_target.view_as(pred)).sum().item()

    test_loss /= len(test_loaders[0].dataset)
    acc = 100.0 * correct / len(test_loaders[0].dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loaders[0].dataset), acc
        )
    )

    # Test upan
    # test_loss, acc = test(config_args, device, upan, upan_test_loader)
    return test_loss, acc


def main(args):

    # Initialize arguments based on testset chosen
    if args.testset == "disjoint_mnist":
        test_loaders = [
            mnist_combined_test_loader(args.test_batch_size),
            mnist_combined_test_loader(args.test_batch_size),
        ]
        args.expert = ["first5_mnist", "last5_mnist"]
        args.input_channel = [1, 1]
        args.output_size = 5
        args.arch = ["resnet18", "resnet18"]
        arch = [ResNet18, ResNet18]
        # m = mnist
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

    # UPAN settings
    if args.upan_type == "logits":
        upan_input_size = args.output_size
        upan_arch = PAN
    elif args.upan_type == "agnostic_logits":
        upan_input_size = 4
        upan_arch = AgnosticPAN

    # Running the test
    print(f"Testset: {args.testset}")
    results = []

    for trial in range(len(args.seeds)):
        seed = args.seeds[trial]
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\nIteration: {trial+1}, Seed: {seed}")

        # Load UPAN model
        upan = upan_arch(input_size=upan_input_size).to(args.device)
        upan.load_state_dict(
            torch.load(
                args.upan_dir
                + f"upan_{args.upan_type}_{args.dataset}{args.arch}_{args.seeds[trial]}",
                map_location=torch.device(args.device),
            )
        )

        all_experts = []
        for expert_idx in range(len(arch)):
            # Load expert
            expert = arch[expert_idx](
                input_channel=args.input_channel[expert_idx], output_size=args.output_size
            ).to(args.device)
            expert.load_state_dict(
                torch.load(
                    args.expert_dir +
                    f"{args.expert[expert_idx]}_{args.arch[expert_idx]}_{args.seeds[trial]}",
                    map_location=torch.device(args.device),
                )
            )
            all_experts.append(expert)

        # Running the experiment
        test_loss, acc = smart_coordinator(args, upan, all_experts, test_loaders)
        result = [{"test_loss": test_loss, "acc": acc}]

        # Adding more info to the result to be saved
        for r in result:
            r.update({"iteration": trial, "seed": args.seeds[trial]})
        results.extend(result)

    # Save the results
    if args.save_results:
        save_results(
            f"upan_{args.upan_type}_{args.dataset}_{args.testset}{args.arch}",
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
        "--upan_type",
        type=str,
        default="agnostic_logits",
        choices=["logits", "agnostic_logits"],
    )
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr, help="learning rate")
    parser.add_argument("--momentum", type=float, default=Config.momentum)
    parser.add_argument("--no_cuda", type=bool, default=Config.no_cuda)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--save_results", type=bool, default=Config.save_results)
    parser.add_argument("--results_dir", type=str, default="./results/merge/")
    parser.add_argument("--expert_dir", type=str, default="./cache/models/")
    parser.add_argument("--upan_dir", type=str, default="./cache/models/upan/")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.seeds = SEEDS
    args.device = device
    main(args)
