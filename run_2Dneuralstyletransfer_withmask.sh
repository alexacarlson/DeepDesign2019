CONTENT_FILE=$1
STYLE_FILE=$2
MASK_FILE=$3
OUTPUT_DIR=$4
IMAGE_SIZE=$5
STYLE_WEIGHT=$6
CONTENT_WEIGHT=$7
NUM_ITERS=$8
#
#
#CONTENT_FILE1=/home/alexandracarlson/Desktop/boulder_images/boulder_depth/Top_zDepth_A.jpeg
#CONTENT_FILE2=None #/home/alexandracarlson/Desktop/floor_plans/modern/modern10.jpg
#MASK_FILE=/home/alexandracarlson/Desktop/boulder_images/boulder_depth/Top_zDepth_A_mask.jpeg
#STYLE_FILE=/home/alexandracarlson/Desktop/ARCH_dec10_2019/baroque2.jpeg
#IMAGE_SIZE=800
#NUM_ITERS=300

cd 2D_to_2D_neural_style_transfer/mask-guided-neural-style

python stylize.py \
     --mask_n_colors=1 \
     --content_img=$CONTENT_FILE1 \
     --target_mask=$MASK_FILE \
     --style_img=$STYLE_FILE \
     --hard_width=$IMAGE_SIZE \
     --iteration ${NUM_ITERS} \
     --style_weight ${STYLE_WEIGHT} \
     --content_weight ${CONTENT_WEIGHT} \
     --output_dir ${OUTPUT_DIR}
