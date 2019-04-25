IMAGE_DATA='/home/alexandracarlson/Desktop/Robotics_Building_networks/Matias_garden_model/images/190319_Basemodel50008.jpg'
#IMAGE_DATA='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'
WEIGHTS_DIR='custom_weights'
RESULTS_DIR='test'
TRAIN_DIR='/mnt/ngv/askc-home/building_motif_dataset/train'

nvidia-docker run --rm -it \
  -v `pwd`:/root \
  -v ${IMAGE_DATA}:${IMAGE_DATA}:ro \
  -v ${TRAIN_DIR}:${TRAIN_DIR}:ro \
  visclass-tf \
  python /root/visualize_class.py \
    --train_vgg_model True \
    --train_epochs 1 \
    --train_dataset ${TRAIN_DIR} \
  	--vgg_model /root/${WEIGHTS_DIR} \
    --dream_image ${IMAGE_DATA} \
    --dream_results_dir /root/results/${RESULTS_DIR} \
  	--image_h 400 \
  	--image_w 600 \
  	--dream_class 'arch'
