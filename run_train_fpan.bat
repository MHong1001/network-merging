call .venv/Scripts/activate

python train_fpan.py --fpan_data=disjoint_mnist --upan_data=disjoint_mnist --upan_type=logits &
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=mnist_cifar10 --upan_type=logits &
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=fmnist_kmnist --upan_type=logits &

python train_fpan.py --fpan_data=disjoint_mnist --upan_data=disjoint_mnist --upan_type=agnostic_logits &
python train_fpan.py --fpan_data=disjoint_mnist --upan_data=mnist_cifar10 --upan_type=agnostic_logits &
python train_fpan.py --fpan_data=disjoint_mnist --upan_data=fmnist_kmnist --upan_type=agnostic_logits &

python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=disjoint_mnist --upan_type=agnostic_logits &
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=mnist_cifar10 --upan_type=agnostic_logits &
python train_fpan.py --fpan_data=mnist_cifar10 --upan_data=fmnist_kmnist --upan_type=agnostic_logits
