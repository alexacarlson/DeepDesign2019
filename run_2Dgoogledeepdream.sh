#
IMAGE_DATA='/storage/2Dmodels/tree1.jpg'
MODEL_DIR='/storage/inception5_weights'
NUM_ITERS=30

python deepdream.py \
  --image_file ${IMAGE_DATA} \
  --network_model ${MODEL_DIR} \
  --iterations ${NUM_ITERS}
