# Towards Neural Network Merging

All the experiments would be ran through 10 distinct random seed. As such, do expect that the application would 
take some time to finish. 

## Requirements 

The software is running with the current libraries, do remember to install them before running.

- Python 3.7 and above
- pytorch 1.4.0
- torchvision 0.5.0
- imgaug 0.2.6
- Pillow 7.0.0

## Training base networks

Training for the base networks can be invoked using the command:

```bash
python train_source_networks.py --dataset=<dataset> --arch=<architecture> --batch_size=<batch size>
```

For the experiments, the following command was used:

```bash
python train_source_networks.py --dataset=first5_mnist --arch=lenet5
python train_source_networks.py --dataset=last5_mnist --arch=lenet5
python train_source_networks.py --dataset=mnist --arch=lenet5
python train_source_networks.py --dataset=cifar10 --arch=resnet18
python train_source_networks.py --dataset=fmnist --arch=resnet18
python train_source_networks.py --dataset=kmnist --arch=resnet18
```

For the full list of arguments please do look into the [source file](./train_source_networks.py).


## Training UPAN

Training for the universal pattern attribution networks(UPAN) for the base networks can be invoked using the command:

```bash
python train_upan.py --dataset=<training set of UPAN> --testset=<problem> --upan_type=<logits, agnostic_logits>
```

Note that this should only be ran after the base network has already been trained.

For the experiments, the following command was used:

```bash
python train_upan.py --dataset=disjoint_mnist –-testset=disjoint_mnist --upan_type=logits
python train_upan.py --dataset=mnist_cifar10 –-testset=mnist_cifar10 --upan_type=logits
python train_upan.py --dataset=mnist_cifar10 –-testset=fmnist_kmnist --upan_type=logits

python train_upan.py --dataset=disjoint_mnist –-testset=disjoint_mnist --upan_type=agnostic_logits
python train_upan.py --dataset=disjoint_mnist –-testset=mnist_cifar10 --upan_type=agnostic_logits
python train_upan.py --dataset=disjoint_mnist –-testset=fmnist_kmnist --upan_type=agnostic_logits

python train_upan.py --dataset=mnist_cifar10 –-testset=disjoint_mnist --upan_type=agnostic_logits
python train_upan.py --dataset=mnist_cifar10 –-testset=mnist_cifar10 --upan_type=agnostic_logits
python train_upan.py --dataset=mnist_cifar10 –-testset=fmnist_kmnist --upan_type=agnostic_logits
```

Here, agnostic logits refers to the logits based activation statistics based upan.

For the full list of arguments please do look into the [source file](./train_upan.py).


## Training FPAN

Training for the fast pattern attribution networks(FPAN) for the base networks can be invoked using the command:

```bash
python train_fpan.py --fpan_data=<training set of FPAN> --upan_data=<training set of UPAN> --upan_type=<logits, agnostic_logits>
```

Note that this should only be ran after the base network and UPAN have already been trained.

For the experiments, the following command was used:

```bash
python train_fpan.py --fpan_data=disjoint_mnist --upan_data=disjoint_mnist --upan_type=logits
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=mnist_cifar10 --upan_type=logits
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=fmnist_kmnist --upan_type=logits

python train_fpan.py --fpan_data=disjoint_mnist --upan_data=disjoint_mnist --upan_type=agnostic_logits
python train_fpan.py --fpan_data=disjoint_mnist --upan_data=mnist_cifar10 --upan_type=agnostic_logits
python train_fpan.py --fpan_data=disjoint_mnist --upan_data=fmnist_kmnist --upan_type=agnostic_logits

python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=disjoint_mnist --upan_type=agnostic_logits
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=mnist_cifar10 --upan_type=agnostic_logits
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=fmnist_kmnist --upan_type=agnostic_logits
```

Here, agnostic logits refers to the logits based activation statistics based upan.

For the full list of arguments please do look into the [source file](./train_fpan.py).


## Running the smart coordinator experiments

The smart coordinator experiments can be run like so:

```bash
python merge_upan.py --dataset=<training set of UPAN> --testset=<problem> -upan_type=<logits,agnostic_logits>
```

Note that the prerequisite for running the experiments must all already exists and prepared.

For the experiments, the following command was used:

```bash
# Logits based upan
python merge_upan.py --dataset=disjoint_mnist --testset=disjoint_mnist --upan_type=logits
python merge_upan.py --dataset=mnist_cifar10 --testset=mnist_cifar10 --upan_type=logits
python merge_upan.py --dataset=mnist_cifar10 --testset=fmnist_kmnist --upan_type=logits

# Logits activation statistics based upan
python merge_upan.py --dataset=disjoint_mnist --testset=disjoint_mnist --upan_type=agnostic_logits
python merge_upan.py --dataset=disjoint_mnist --testset=mnist_cifar10 --upan_type=agnostic_logits
python merge_upan.py --dataset=disjoint_mnist --testset=fmnist_kmnist --upan_type=agnostic_logits
python merge_upan.py --dataset=mnist_cifar10 --testset=disjoint_mnist --upan_type=agnostic_logits
python merge_upan.py --dataset=mnist_cifar10 --testset=mnist_cifar10 --upan_type=agnostic_logits
python merge_upan.py --dataset=mnist_cifar10 --testset=fmnist_kmnist --upan_type=agnostic_logits
```

For the full list of arguments please do look into the [source file](./merge_upan.py).
