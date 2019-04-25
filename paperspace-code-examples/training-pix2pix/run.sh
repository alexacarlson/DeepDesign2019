#! /bin/bash
export PYTHONUNBUFFERED=0
python pix2pix.py \
  --mode train \
  --output_dir /artifacts \
  --max_epochs 200 \
  --input_dir images/combined/train \
  --which_direction BtoA \
  --save_freq 100 \
  --display_freq 150 \
