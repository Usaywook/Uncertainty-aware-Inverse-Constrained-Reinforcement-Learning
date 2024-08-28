#!/bin/bash

DEVICE=2

# train ICRL
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s 123
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s 321
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s 456
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s 654
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_ICRL_HC-noise-1e-1.yaml -n 5 -s 666

