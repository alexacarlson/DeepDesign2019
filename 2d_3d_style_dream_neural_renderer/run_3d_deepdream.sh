#INPUT_OBJ_PATH='TreeCartoon/TreeCartoon1_OBJ/TreeCartoon1_OBJ.obj'
#OUTPUT_FILENAME='2Dtree_3Dtree.gif'
#INPUT_OBJ_PATH=Alexa/190502_Dreamground_D.obj

FILE_NAME_OBJ=190730_Rock_A
#FILE_NAME_OBJ=190730_Rock_B
#FILE_NAME_OBJ=190730_Rock_C
#FILE_NAME_OBJ=190730_Rock_D
#FILE_NAME_OBJ=190730_Rock_E
#
#FILE_NAME_OBJ=190514_Dreaming_Ground_A_1k
#OBJ_DIR=190514_Dreaming_Ground #190515_components
#FILE_NAME_OBJ=190513_ground_C
#FILE_NAME_OBJ=1905013_Dreaming_Ground_A.obj
#FILE_NAME_OBJ=1905013_Dreaming_Ground_B.obj
#FILE_NAME_OBJ=1905013_Dreaming_Ground_C.obj
#FILE_NAME_OBJ=1905013_Dreaming_Ground_D.obj
#FILE_NAME_OBJ=1905013_Dreaming_Ground_E.obj
#INPUT_OBJ_PATH=${OBJ_DIR}/${FILE_NAME_OBJ}.obj
INPUT_OBJ_PATH=${FILE_NAME_OBJ}.obj
NUM_ITER=5
OUTPUT_DIR=TEST/${FILE_NAME_OBJ}
OUTPUT_FILENAME=${OUTPUT_DIR}/${FILE_NAME_OBJ}_dreamed.gif
#OUTPUT_FILENAME=${FILE_NAME_OBJ}_dreamed.gif
IMAGE_SIZE=1024
DIR_LOC=/mnt/ngv/askc-home/ARCHITECTURE_ROBOTICSGARDEN_PROJECT/Robotics_Building_networks

# 2d3d-dreamstyle
nvidia-docker run --rm -it \
	-v ${DIR_LOC}/3D_models_newboulders/:/3Dmodels:ro \
	-v ${DIR_LOC}/2Dmodels:/2Dmodels:ro \
	-v `pwd`:/root \
	acarlson32/2d3d_neuralrenderer:firstimage \
python run_examples_deep_dream_3d/run.py \
    --filename_obj /3Dmodels/${INPUT_OBJ_PATH} \
    --output_directory /root/${OUTPUT_DIR} \
    --filename_output /root/${OUTPUT_FILENAME} \
    --num_iteration ${NUM_ITER} \
    --image_size ${IMAGE_SIZE} \
    --gpu 0
