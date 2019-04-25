IMAGE_DATA='/home/alexandracarlson/Desktop/Robotics_Building_networks/Matias_garden_model/images/190319_Basemodel50008.jpg'
#IMAGE_DATA='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'
WEIGHTS_DIR='custom_weights'
RESULTS_DIR='test'

nvidia-docker run --rm -it \
  -v `pwd`:/root \
  -v ${IMAGE_DATA}:${IMAGE_DATA}:ro \
  visclass-tf \
  python /root/visualize_class.py \
  	--vgg_model /root/${WEIGHTS_DIR} \
    --dream_image ${IMAGE_DATA} \
    --dream_results_dir /root/results/${RESULTS_DIR} \
  	--image_h 400 \
  	--image_w 600 \
  	--dream_class 'arch'
