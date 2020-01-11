#IMAGE_DATA='/storage/2Dmodels/scene0_camloc_0_5_-20_rgb.png'
#WEIGHTS_DIR='/storage/acadia_general_arch_styles_netweights'
#RESULTS_DIR='/storage/test'
#NUM_ITERS=500
#DREAM_CLASS='gothic'

IMAGE_DATA=$1
WEIGHTS_DIR=$2
DREAM_CLASS=$3
RESULTS_DIR=$4
NUM_ITERS=$5
IMAGE_H=$6
IMAGE_W=$7

cd 2D_class_based_dreaming
#visclass-tf \
python visualize_class.py \
	--vgg_model ${WEIGHTS_DIR} \
	--dream_image ${IMAGE_DATA} \
	--dream_results_dir ${RESULTS_DIR} \
	--image_h ${IMAGE_H} \
	--image_w ${IMAGE_W} \
	--dream_class ${DREAM_CLASS} \
	--iterations ${NUM_ITERS}
