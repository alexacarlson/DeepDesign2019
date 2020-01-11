PHASE=training
ROOTFOLDER=/mnt/ngv/
DATASET=${ROOTFOLDER}/datasets/Snow100k-desnownetpaperdataset
TFRECORD_DIR=/mnt/ngv/askc-home/SYNTH_TO_REAL_DOMAINADAPT_LEARNINGNATURALIMAGEMANIFOLD/tfrecord_pggan_${PHASE}
IMAGE_DIR=/mnt/ngv/askc-home/SYNTH_TO_REAL_DOMAINADAPT_LEARNINGNATURALIMAGEMANIFOLD/all_files_oxfkittirawcitybdd_list.txt
#
## run in docker
nvidia-docker run --rm -it \
	-v ${ROOTFOLDER}/datasets/bdd100k:${ROOTFOLDER}/datasets/bdd100k \
	-v ${ROOTFOLDER}/askc-home:${ROOTFOLDER}/askc-home \
	-v `pwd`/:/root \
	datmo/keras-tensorflow:gpu-py35 \
	python /root/dataset_tool.py \
		create_from_imagedir \
		 ${TFRECORD_DIR}\
		 ${IMAGE_DIR}

#		 ${DATASET}/${PHASE}/gt_512x512
