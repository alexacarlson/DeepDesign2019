CONTENT_FILE=$1
STYLE_FILE=$2
OUTPUT_FILE=$3
IMAGE_SIZE=$4

nvidia-docker run --rm -it \
  -v `pwd`:/root \
  neuralstyle-tf \
  python /root/neural_style.py \
  --content $CONTENT_FILE \
  --styles $STYLE_FILE \
  --output $OUTPUT_FILE \
  --width $IMAGE_SIZE

