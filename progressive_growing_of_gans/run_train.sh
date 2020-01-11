#ROOTFOLDER=/mnt/ngv/askc-home/WEATHERTRANSFER/
#NUM_EPOCHS=100
#PHASE='train'
#GPU_ID='3'
BUILD_MAIN_PY=train_snow100k
DATASET=/mnt/ngv/askc-home/SYNTH_TO_REAL_DOMAINADAPT_LEARNINGNATURALIMAGEMANIFOLD/tfrecord_pggan_training/
#
## run in docker
nvidia-docker run --rm -it \
	-v ${DATASET}:/datasets/alldata \
	-v `pwd`/:/root \
	datmo/keras-tensorflow:gpu-py35 \
	python /root/${BUILD_MAIN_PY}.py 

2>&1 | tee -a output-logs.txt
