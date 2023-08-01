#!/bin/bash
#project="optim-robust-cifar10-cifar100-svhn"
#project="optim-robust-tiny-dtd-imagenette"
#project="ood-preactresnet18"
#project="ood-preactresnet50"
project="tmlr_remove_high_freq_low_amp_mar24"
#gpu="t4v2,rtx6000"
gpu="t4v2"
#gpu="t4v1,p100,t4v2"
enable_wandb=true #true/false

eval_PGD=true
eval_AA=false
eval_CC=false
eval_TMLR=false
pgd_clip=true

standard_DA=true

lr_update='multistep'
method='standard'
date=`date +%Y%m%d`
epoch=200
arch='preactresnet18'
input_normalization=false
enable_batchnorm=false
momentum=0
optim='sgd'

batch_size=32

#dataset='cifar10'
#lr=0.2
#dataset='cifar100'
#lr=0.3
#dataset='svhn'
#lr=0.2
dataset='caltech'
lr=0.05
#dataset='imagenette'
#lr=0.1

for seed in 40 41 42 43; do
	for method in 'remove_high_freq' 'remove_low_amp'; do
		for threshold in 10 30 50 70 90; do
	#for method in 'standard'; do
		#for threshold in 0; do
			j_name=${date}'-'${dataset}'-'${method}'-'${optim}'-'${threshold}'-'${seed}'-'$RANDOM
			bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 main.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --arch \"${arch}\" --seed ${seed} --eval_PGD ${eval_PGD} --eval_AA ${eval_AA} --eval_CC ${eval_CC} --optim ${optim} --momentum ${momentum} --lr_update \"${lr_update}\" --standard_DA ${standard_DA} --pgd_clip ${pgd_clip} --lambbda 0 --input_normalization ${input_normalization} --enable_batchnorm ${enable_batchnorm} --batch_size ${batch_size} --adam_beta1 0.9 --adam_beta2 0.999 --rmsp_alpha 0.99 --threshold ${threshold} --eval_TMLR ${eval_TMLR}" 
			sleep 0.1
		done
	done
done
