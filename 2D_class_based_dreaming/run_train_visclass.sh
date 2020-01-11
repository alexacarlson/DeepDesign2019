WEIGHTS_DIR='sam_mandelbrot_classes_netweights'
RESULTS_DIR='sam_mandelbrot_classes_results'
TRAIN_DIR='/mnt/ngv/askc-home/ARCHITECTURE_ROBOTICSGARDEN_PROJECT/mandelbrot_dataset_sam'
TRAIN_EPOCHS=1
IMAGE_H=900
IMAGE_W=900

python /root/train_class_vgg.py \
  --train_epochs ${TRAIN_EPOCHS} \
  --train_dataset ${TRAIN_DIR} \
	--vgg_model ${WEIGHTS_DIR} \
  --results_dir ${RESULTS_DIR} \
	--image_h ${IMAGE_H} \
	--image_w ${IMAGE_W} 