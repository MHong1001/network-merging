call .venv/Scripts/activate

REM Logits based pan
python merge_upan.py --dataset=disjoint_mnist --testset=disjoint_mnist --upan_type=logits &
python merge_upan.py --dataset=mnist_cifar10 --testset=mnist_cifar10 --upan_type=logits &
python merge_upan.py --dataset=mnist_cifar10 --testset=fmnist_kmnist --upan_type=logits &

REM agnostic logits based pan
python merge_upan.py --dataset=disjoint_mnist --testset=disjoint_mnist --upan_type=agnostic_logits &
python merge_upan.py --dataset=disjoint_mnist --testset=mnist_cifar10 --upan_type=agnostic_logits &
python merge_upan.py --dataset=disjoint_mnist --testset=fmnist_kmnist --upan_type=agnostic_logits &
python merge_upan.py --dataset=mnist_cifar10 --testset=disjoint_mnist --upan_type=agnostic_logits &
python merge_upan.py --dataset=mnist_cifar10 --testset=mnist_cifar10 --upan_type=agnostic_logits &
python merge_upan.py --dataset=mnist_cifar10 --testset=fmnist_kmnist --upan_type=agnostic_logits