#!/bin/bash
gpu="a40,t4v2,rtx6000"

for dataset in 'mnist' 'fashionmnist'; do
# for dataset in 'cifar10' 'cifar100' 'svhn'; do
# for dataset in 'imagenette' 'caltech'; do
# for dataset in 'cifar10' 'cifar100' 'svhn' 'imagenette' 'caltech'; do
    # for norm in 'L2'; do
    for norm in 'Linf'; do
        # for eps in 0.1 0.2 0.3; do
        # for eps in 0.1 0.5 1.0 1.5 2.0; do
        for eps in 0.005 0.01 0.05 0.1; do
        # for eps in 0.00392 0.00784 0.01568; do
        # for eps in 0.1 0.05 0.01; do
            j_name=${dataset}'-'${norm}'-'${eps}
            bash launch_eval_AA_cluster.sh ${gpu} ${j_name} 1 "python3 eval_AA_cluster.py --norm \"${norm}\" --dataset \"${dataset}\" --eps ${eps}" 
        done
    done
done
