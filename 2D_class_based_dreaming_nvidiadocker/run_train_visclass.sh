SINGLE_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/Matias_garden_model/images/190319_Basemodel50008.jpg'
MULTI_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'

TRAIN_DIR='/mnt/ngv/askc-home/building_motif_dataset/train'
WEIGHTS_DIR='custom_weights2'

nvidia-docker run --rm -it \
  -v `pwd`:/root \
  -v ${TRAIN_DIR}:${TRAIN_DIR}:ro \
  visclass-tf \
  python /root/visualize_class.py \
    --train_vgg_model False \
    --train_epochs 1 \
    --train_dataset ${TRAIN_DIR} \
  	--vgg_model /root/${WEIGHTS_DIR} \
  	--image_h 400 \
  	--image_w 600 \
  	--dream_class 'arch'
