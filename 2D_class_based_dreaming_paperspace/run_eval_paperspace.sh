SINGLE_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/Matias_garden_model/images/190319_Basemodel50008.jpg'
MULTI_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'
WEIGHTS_DIR='custom_weights_vgg16'
RESULTS_DIR='results_dreaming_with_custom_weights'

#visclass-tf \
python visualize_class.py \
  	--vgg_model /storage/${WEIGHTS_DIR} \
  	--dream_image ${SINGLE_IMAGE_DIR} \
	--dream_results_dir ${RESULTS_DIR} \
	--image_h 400 \
  	--image_w 600 \
  	--dream_class 'arch'
