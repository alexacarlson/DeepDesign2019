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
python visualize_class_gpu2.py \
	--vgg_model ${WEIGHTS_DIR} \
	--dream_image ${IMAGE_DATA} \
	--dream_results_dir ${RESULTS_DIR} \
	--image_h ${IMAGE_H} \
	--image_w ${IMAGE_W} \
	--dream_class ${DREAM_CLASS} \
	--iterations ${NUM_ITERS}
	
#
#IMAGE_DATA=/storage/Boulder_Grey
#WEIGHTS_DIR=`pwd`/sam_mandelbrot_classes_netweights
#DREAMED_CLASS='18PRS11_08_BCN_Pres_MandlbTomography_02'
#RESULTS_DIR=/home/alexandracarlson/Desktop/boulder_images/dreamedclass${DREAMED_CLASS}_${IMAGE_DATA_NAME}
#IMAGE_H=900
#IMAGE_W=900
#NUM_ITERS=50
#
#python /root/visualize_class_gpu2.py \
#  --vgg_model ${WEIGHTS_DIR} \
#  --dream_image ${IMAGE_DATA} \
#  --iterations ${NUM_ITERS} \
#  --dream_class ${DREAMED_CLASS} \
#  --dream_results_dir ${RESULTS_DIR} \
#  --image_h ${IMAGE_H} \
#  --image_w ${IMAGE_W} \
