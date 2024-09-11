#!/bin/bash

DEVICE=6
SEEDS=(123 321 456 654 666)

for SEED in ${SEEDS[@]}
do
    #################################################
    # highD_velocity_constraint with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/highD/velocity_constraint/train_ppo_lag_highD_velocity_constraint.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_constraint/train_Binary_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/highD/velocity_constraint/train_GAIL_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_constraint/train_ICRL_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_constraint/train_VICRL_highD_velocity_constraint1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_constraint/train_DICRL_QRDQN_CVaR_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/highD/velocity_constraint/train_DICRL_QRDQN_EXP_highD_velocity_constraint-1e-1.yaml -n 5 -s $SEED

    #################################################
    # HCWithPos-v0 with 0.1 noise
    #################################################

    # run PPO_Lag
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_PPO-Lag_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run DLPO
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_DLPO-Averse_HC-noise-1e-1.yaml -n 5 -s $SEED
    CUDA_VISIBLE_DEVICES=$DEVICE python train_policy.py ../config/Mujoco/Blocked_HalfCheetah/train_DLPO-Neutral_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run BC2L
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_BC2L_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run ICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run VICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_VICRL_HC-noise-1e-1.yaml -n 5 -s $SEED

    # run UAICRL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_UAICRL_HC-noise-1e-1.yaml -n 5 -s $SEED

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
