IMAGE_DATA='/storage/2Dmodels/scene0_camloc_0_5_-20_rgb.png'
WEIGHTS_DIR='/storage/acadia_general_arch_styles_netweights'
#RESULTS_DIR='/artifacts/test'
RESULTS_DIR='/storage/test'
NUM_ITERS=500

#visclass-tf \
python visualize_class.py \
	--vgg_model ${WEIGHTS_DIR} \
    --dream_image ${IMAGE_DATA} \
    --dream_results_dir ${RESULTS_DIR} \
	--image_h 720 \
	--image_w 1280 \
	--dream_class 'gothic' \
	--iterations ${NUM_ITERS}
