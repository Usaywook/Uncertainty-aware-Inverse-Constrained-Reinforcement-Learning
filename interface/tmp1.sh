#!/bin/bash

DEVICE=6
SEEDS=(123 321 456)
# train UAICRL

for SEED in ${SEEDS[@]}
do
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/HighD_velocity_constraint/train_DICRL_QRDQN_CVaR_highD_velocity_constraint.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/HighD_velocity_constraint/train_DICRL_QRDQN_EXP_highD_velocity_constraint.yaml -n 5 -s $SEED
done

