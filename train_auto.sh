#!/usr/bin/env bash

NUM_STEPS=100000
mkdir -p checkpoints


CHAIRS_URL="https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip"
CHAIRS_ZIP=$(basename $CHAIRS_URL)

if [[ ! -f "$CHAIRS_ZIP"  ]]; then
  wget "$CHAIRS_URL"
fi

if [[ ! -d "./datasets/FlyingChairs_release" ]]; then
  unzip -q "$CHAIRS_ZIP" -d datasets
fi



echo "Training on chairs"

python -u train.py --name raft-chairs --stage chairs --validation chairs \
  --mixed_precision \
  --tiny \
  --gpus 0 \
  --num_steps ${NUM_STEPS} \
  --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001




SINTEL_URL="http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
SINTEL_ZIP=$(basename $SINTEL_URL)

if [[ ! -f "$SINTEL_ZIP"  ]]; then
  wget "$SINTEL_URL"
fi

if [[ ! -d "./datasets/sintel" ]]; then
  unzip -q "$SINTEL_ZIP" -d datasets/Sintel
fi

echo "Training on sintel"

python -u train.py --name raft-sintel --stage sintel --validation sintel \
  --mixed_precision \
  --tiny \
  --gpus 0 \
  --num_steps ${NUM_STEPS} \
  --restore_ckpt checkpoints/raft-chairs.pth \
  --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85