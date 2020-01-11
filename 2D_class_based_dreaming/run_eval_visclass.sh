#
IMAGE_DATA=/storage/Boulder_Grey
WEIGHTS_DIR=`pwd`/sam_mandelbrot_classes_netweights
DREAMED_CLASS='18PRS11_08_BCN_Pres_MandlbTomography_02'
RESULTS_DIR=/home/alexandracarlson/Desktop/boulder_images/dreamedclass${DREAMED_CLASS}_${IMAGE_DATA_NAME}
IMAGE_H=900
IMAGE_W=900
NUM_ITERS=50

python /root/visualize_class_gpu2.py \
  --vgg_model ${WEIGHTS_DIR} \
  --dream_image ${IMAGE_DATA} \
  --iterations ${NUM_ITERS} \
  --dream_class ${DREAMED_CLASS} \
  --dream_results_dir ${RESULTS_DIR} \
  --image_h ${IMAGE_H} \
  --image_w ${IMAGE_W} \

#  -v ${IMAGE_DATA}:${IMAGE_DATA}:ro \
