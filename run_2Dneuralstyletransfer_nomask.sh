CONTENT_FILE=$1
STYLE_FILE=$2
OUTPUT_FILE=$3
IMAGE_SIZE=$4
STYLE_WEIGHT=$5
CONTENT_WEIGHT=$6
#
#CONTENT_FILE1=/home/alexandracarlson/Desktop/boulder_images/boulder_depth/Side4_zDepth_A.jpeg
#STYLE_FILE=/home/alexandracarlson/Desktop/boulder_images/style_guide_images/shark_depth.jpeg
#IMAGE_SIZE=800
#NUM_ITERS=100
#STYLE_WEIGHT=0.5
#CONTENT_WEIGHT=1.0

cd 2D_to_2D_neural_style_transfer/mask-guided-neural-style

python stylize.py \
     --mask_n_colors=0 \
     --content_img=$CONTENT_FILE1 \
     --style_img=$STYLE_FILE \
     --hard_width=$IMAGE_SIZE \
     --iteration ${NUM_ITERS} \
     --style_weight ${STYLE_WEIGHT} \
     --content_weight ${CONTENT_WEIGHT} 
