#IMAGE_DATA='/storage/scene0_camloc_0_5_-20_rgb.png'
#WEIGHTS_DIR='custom_weights_vgg16_train'
#RESULTS_DIR='/storage/test_train'
#TRAIN_DIR='/mnt/ngv/askc-home/building_motif_dataset/train'

TRAIN_DIR=$1
TRAIN_EPOCHS=$2
WEIGHTS_DIR=$3
IMAGE_H=$4
IMAGE_W=$5

cd 2D_class_based_dreaming

#visclass-tf \
python visualize_class_gpu2.py \
  --train_vgg_model True \
  --train_epochs ${TRAIN_EPOCHS} \
  --train_dataset ${TRAIN_DIR} \
  --vgg_model ${WEIGHTS_DIR} \
  --image_h ${IMAGE_H} \
  --image_w ${IMAGE_W} \
