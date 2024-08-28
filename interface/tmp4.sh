#!/bin/bash

DEVICE=3

# train GAIL
CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s 123
CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s 321
CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s 456
CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s 654
CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/Mujoco/Blocked_HalfCheetah/train_GAIL_HC-noise-1e-1.yaml -n 5 -s 666

