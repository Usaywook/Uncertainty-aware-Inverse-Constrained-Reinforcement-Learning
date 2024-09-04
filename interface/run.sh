#!/bin/bash

DEVICE=0
SEEDS=(123 321 456 654 666)

for SEED in ${SEEDS[@]}
do
    #################################################
    # highD_velocity_constraint with 0.1 noise
    #################################################

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/highD_velocity_constraint/train_GAIL_highd_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD_velocity_constraint/train_Binary_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    # run MECL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD_velocity_constraint/train_ICRL_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD_velocity_constraint/train_VICRL_highD_velocity_constraint1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD_velocity_constraint/train_DICRL_QRDQN_CVaR_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD_velocity_constraint/train_DICRL_QRDQN_EXP_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

done
