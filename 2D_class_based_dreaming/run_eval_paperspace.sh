IMAGE_DATA='/storage/scene0_camloc_0_5_-20_rgb.png'
#IMAGE_DATA='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'
WEIGHTS_DIR='custom_weights_vgg16'
#RESULTS_DIR='/artifacts/test'
RESULTS_DIR='/storage/test'

#visclass-tf \
python visualize_class.py \
	--vgg_model /storage/${WEIGHTS_DIR} \
  --dream_image ${IMAGE_DATA} \
  --dream_results_dir ${RESULTS_DIR} \
	--image_h 400 \
	--image_w 600 \
	--dream_class 'arch'
