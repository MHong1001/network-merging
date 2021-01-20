call .venv/Scripts/activate

python train_source_networks.py --dataset=first5_mnist --arch=resnet18
python train_source_networks.py --dataset=last5_mnist --arch=resnet18

python train_source_networks.py --dataset=mnist --arch=resnet18
python train_source_networks.py --dataset=cifar10 --arch=resnet18

python train_source_networks.py --dataset=fmnist --arch=resnet18
python train_source_networks.py --dataset=kmnist --arch=resnet18

REM python train_source_networks.py --dataset=first5_mnist &
REM python train_source_networks.py --dataset=last5_mnist &
REM python train_source_networks.py --dataset=mnist &
REM python train_source_networks.py --dataset=cifar10 --batch_size=128 &

REM python train_source_networks.py --dataset=mnist --arch=resnet18 --lr=0.1 &
REM python train_source_networks.py --dataset=cifar10 --arch=resnet18 --batch_size=128 --lr=0.1 --epochs 20 

REM python train_source_networks.py --dataset=first5_mnist --arch=lenet5_halfed &
REM python train_source_networks.py --dataset=last5_mnist --arch=lenet5_halfed &
REM python train_source_networks.py --dataset=mnist --arch=lenet5_halfed &
REM python train_source_networks.py --dataset=cifar10 --arch=lenet5_halfed --batch_size=128