#!/bin/bash

DEVICE=6
SEEDS=(123 321 456)

for SEED in ${SEEDS[@]}
do
    #################################################
    # highD_distance_constraint with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/distance_constraint/train_ppo_lag_highD_distance_constraint.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/distance_constraint/train_DLPO_QRDQN-Averse_highD_distance_constraint.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/distance_constraint/train_DLPO_QRDQN-Neutral_highD_distance_constraint.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/distance_constraint/train_Binary_highD_distance_constraint.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/highD/distance_constraint/train_GAIL_highD_distance_constraint.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/distance_constraint/train_ICRL_highD_distance_constraint.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/distance_constraint/train_VICRL_highD_distance_constraint.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/distance_constraint/train_DICRL_QRDQN_CVaR_highD_distance_constraint.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/distance_constraint/train_DICRL_QRDQN_EXP_highD_distance_constraint.yaml -n 5 -s $SEED

    #################################################
    # highD_velocity_distance_constraint with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_distance_constraint/train_ppo_lag_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_distance_constraint/train_DLPO_QRDQN-Averse_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_distance_constraint/train_DLPO_QRDQN-Neutral_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_distance_constraint/train_Binary_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/highD/velocity_distance_constraint/train_GAIL_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_distance_constraint/train_ICRL_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_distance_constraint/train_VICRL_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_distance_constraint/train_DICRL_QRDQN_CVaR_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_distance_constraint/train_DICRL_QRDQN_EXP_highD_velocity_distance_constraint.yaml -n 5 -s $SEED
done
