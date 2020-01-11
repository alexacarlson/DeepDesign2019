#
#IMAGE_DATA='/storage/2Dmodels/tree1.jpg'
#MODEL_DIR='/storage/inception5_weights/tensorflow_inception_graph.pb'
#NUM_ITERS=30

IMAGE_DATA=$1
MODEL_DIR=$2
RESULTS_DIR=$3
NUM_ITERS=$4

pip install pillow
cd google_DeepDreaming/

python deepdream.py \
  --image_file ${IMAGE_DATA} \
  --network_model ${MODEL_DIR} \
  --dream_results_dir ${RESULTS_DIR} \
  --iterations ${NUM_ITERS}
