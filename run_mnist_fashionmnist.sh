#!/bin/bash
project="tmlr_remove_high_freq_low_amp_mar24"
#gpu="t4v2,rtx6000"
#gpu="a40"
gpu="t4v2"
enable_wandb=true #true/false

eval_PGD=true
eval_AA=false
eval_CC=false
eval_TMLR=false

lr_update='multistep'

arch='c2'
input_d=28

input_d=28

hidden_d=1
output_d=10
weight_init='kaiming_normal'
bias=false
freq=false
pgd_clip=true

date=`date +%Y%m%d`
epoch=200
momentum=0.
optim='sgd'

dataset='mnist'
lr=0.1
#dataset='fashionmnist'
#lr=0.01

for seed in 41 42 43 44; do
	for method in 'remove_high_freq' 'remove_low_amp'; do
		for threshold in 10 30 50 70 90; do
	#for method in 'standard'; do
		#for threshold in 0; do
			j_name=${date}'-'${dataset}'-'${method}'-'${optim}'-'${threshold}'-'${seed}'-'$RANDOM
			bash launch_slurm_job.sh ${gpu} ${j_name} 1 "python3 main.py --method \"${method}\" --lr ${lr} --dataset \"${dataset}\" --epoch ${epoch} --wandb_project \"${project}\" --enable_wandb ${enable_wandb} --arch \"${arch}\" --seed ${seed} --eval_PGD ${eval_PGD} --eval_AA ${eval_AA} --eval_CC ${eval_CC} --optim ${optim} --momentum ${momentum} --lr_update \"${lr_update}\" --standard_DA 0 --input_d ${input_d} --hidden_d ${hidden_d} --output_d ${output_d} --weight_init \"${weight_init}\" --bias ${bias} --activation relu --freq ${freq} --pgd_clip ${pgd_clip} --lambbda 0 --adam_beta1 0.9 --adam_beta2 0.999 --rmsp_alpha 0.99 --threshold ${threshold} --eval_TMLR ${eval_TMLR}"
			sleep 0.1
		done
	done
done
