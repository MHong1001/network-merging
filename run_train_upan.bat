call .venv/Scripts/activate

python train_upan.py --dataset=disjoint_mnist --testset=disjoint_mnist --upan_type=logits &
python train_upan.py --dataset=mnist_cifar10 --testset=mnist_cifar10 --upan_type=logits &
python train_upan.py --dataset=mnist_cifar10 --testset=fmnist_kmnist --upan_type=logits &

python train_upan.py --dataset=disjoint_mnist --testset=disjoint_mnist --upan_type=agnostic_logits &
python train_upan.py --dataset=disjoint_mnist --testset=mnist_cifar10 --upan_type=agnostic_logits &
python train_upan.py --dataset=disjoint_mnist --testset=fmnist_kmnist --upan_type=agnostic_logits &

python train_upan.py --dataset=mnist_cifar10 --testset=disjoint_mnist --upan_type=agnostic_logits &
python train_upan.py --dataset=mnist_cifar10 --testset=mnist_cifar10 --upan_type=agnostic_logits &
python train_upan.py --dataset=mnist_cifar10 --testset=fmnist_kmnist --upan_type=agnostic_logits