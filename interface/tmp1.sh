#!/bin/bash

DEVICE=6
SEEDS=(123 321 456)

for SEED in ${SEEDS[@]}
do
    #################################################
    # highD_velocity_constraint with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_constraint/train_ppo_lag_highD_velocity_constraint.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_constraint/train_DLPO_QRDQN-Averse_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_constraint/train_DLPO_QRDQN-Neutral_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    #################################################
    # HCWithPos-v0 with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_PPO-Lag_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_DLPO-Averse_HC-noise-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_DLPO-Neutral_HC-noise-1e-1.yaml -n 5 -s $SEED

    #################################################
    # AntWall-v0 with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Ant/train_PPO-Lag_Ant-noise-1e-1.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Ant/train_DLPO-Averse_Ant-noise-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Ant/train_DLPO-Neutral_Ant-noise-1e-1.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Ant/train_BC2L_Ant-noise-1e-1.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_Ant/train_GAIL_Ant-noise-1e-1.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Ant/train_ICRL_Ant-noise-1e-1.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Ant/train_VICRL_Ant-noise-1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Ant/train_UAICRL_Ant-noise-1e-1.yaml -n 5 -s $SEED

done
