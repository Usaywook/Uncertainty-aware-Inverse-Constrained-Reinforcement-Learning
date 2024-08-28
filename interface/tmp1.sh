#!/bin/bash

DEVICE=0

# train UAICRL
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_UAICRL_HC-noise-1e-1.yaml -n 5 -s 321
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_UAICRL_HC-noise-1e-1.yaml -n 5 -s 456
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_UAICRL_HC-noise-1e-1.yaml -n 5 -s 654
CUDA_VISIBLE_DEVICES=$DEVICE python train_icrl.py ../config/Mujoco/Blocked_HalfCheetah/train_UAICRL_HC-noise-1e-1.yaml -n 5 -s 666
