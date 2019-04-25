#!/bin/bash

APT_PACKAGES="apt-utils ffmpeg libav-tools x264 x265"
apt-install() {
	export DEBIAN_FRONTEND=noninteractive
	apt-get update -q
	apt-get install -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" $APT_PACKAGES
	return $?
}

#install ffmpeg to container
add-apt-repository -y ppa:jonathonf/ffmpeg-3 2>&1
apt-install || exit 1

#create folders
mkdir data
mkdir data/bin

#run style transfer on video
python transform_video.py --in-path examples/content/fox.mp4 \
  --checkpoint ./scream.ckpt \
  --out-path /artifacts/out.mp4 \
  --device /gpu:0 \
  --batch-size 4 2>&1
