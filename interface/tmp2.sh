#!/bin/bash

DEVICE=6
SEEDS=(123 321 456)

for SEED in ${SEEDS[@]}
do
    #################################################
    # InvertedPendulumWall-v0 with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Biased_Pendulum/train_PPO-Lag_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Biased_Pendulum/train_DLPO-Averse_Pendulum-noise-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Biased_Pendulum/train_DLPO-Neutral_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Biased_Pendulum/train_BC2L_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Biased_Pendulum/train_GAIL_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Biased_Pendulum/train_ICRL_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Biased_Pendulum/train_VICRL_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Biased_Pendulum/train_UAICRL_Pendulum-noise-1e-1.yaml -n 5 -s $SEED

    #################################################
    # WalkerWithPos-v0 with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Walker/train_PPO-Lag_Walker-noise-1e-1.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Walker/train_DLPO-Averse_Walker-noise-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Walker/train_DLPO-Neutral_Walker-noise-1e-1.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Walker/train_BC2L_Walker-noise-1e-1.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_Walker/train_GAIL_Walker-noise-1e-1.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Walker/train_ICRL_Walker-noise-1e-1.yamll -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Walker/train_VICRL_Walker-noise-1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Walker/train_UAICRL_Walker-noise-1e-1.yaml -n 5 -s $SEED

    #################################################
    # SwimmerWithPos-v0 with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Swimmer/train_PPO-Lag_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Swimmer/train_DLPO-Averse_Swimmer-noise-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_Swimmer/train_DLPO-Neutral_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Swimmer/train_BC2L_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_Swimmer/train_GAIL_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Swimmer/train_ICRL_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Swimmer/train_VICRL_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_Swimmer/train_UAICRL_Swimmer-noise-1e-1.yaml -n 5 -s $SEED

done
