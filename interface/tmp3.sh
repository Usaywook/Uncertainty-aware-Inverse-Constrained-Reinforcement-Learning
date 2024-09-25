#!/bin/bash

DEVICE=6
SEEDS=(123)

for SEED in ${SEEDS[@]}
do
    #################################################
    # highD_velocity_distance_constraint with 0.1 noise
    #################################################

    # run GACL
    CUDA_VISIBLE_DEVICES=$DEVICE python train_gail.py ../config/highD/velocity_distance_constraint/train_GAIL_highD_velocity_distance_constraint.yaml -n 5 -s $SEED

done
