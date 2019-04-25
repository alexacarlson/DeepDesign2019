#neuralstyle-tf \
python neural_style.py \
  --network /storage/imagenet-vgg-verydeep-19.mat \
  --content boulder.png \
  --styles test_images/skateparkbench.png \
  --output /artifacts/boulder_style.png \
  --width 400

