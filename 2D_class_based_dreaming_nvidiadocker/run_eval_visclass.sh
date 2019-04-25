SINGLE_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/Matias_garden_model/images/190319_Basemodel50008.jpg'
MULTI_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'
WEIGHTS_DIR='custom_weights'

nvidia-docker run --rm -it \
  -v `pwd`:/root \
  -v ${SINGLE_IMAGE_DIR}:/root/single_image.jpg:ro \
  -v ${MULTI_IMAGE_DIR}:/root/multi_images:ro \
  -v `pwd`/building_motif_dataset/train:/building_motif_dataset/train:ro \
  visclass-tf \
  python /root/visualize_class.py \
  	--vgg_model /root/${WEIGHTS_DIR} \
  	--image_h 400 \
  	--image_w 600 \
  	--dream_class 'arch'
