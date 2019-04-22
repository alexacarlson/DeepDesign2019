SINGLE_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/Matias_garden_model/images/190319_Basemodel50008.jpg'
MULTI_IMAGE_DIR='/home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/templeRing/images_rot'

nvidia-docker run --rm -it \
  -v `pwd`:/root \
  -v ${SINGLE_IMAGE_DIR}:/root/single_image.jpg:ro \
  -v ${MULTI_IMAGE_DIR}:/root/multi_images:ro \
  -v `pwd`/building_motif_dataset/train:/building_motif_dataset/train:ro \
  visclass-tf \
  python /root/visualize_class.py \
#  --trainer MUNIT\
#  --config /root/configs/munit_cornellbox_folder.yaml\
#  --output_path /outputs \
#  --gpu_id 2 \#

#  2>&1 | tee -a st-image2image-Munit-cornellbox-logs.txt