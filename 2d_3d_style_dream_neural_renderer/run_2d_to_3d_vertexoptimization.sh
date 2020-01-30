#INPUT_OBJ_PATH='Alexa/190502_Dreamground_A.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_B.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_C.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_D.obj'
#
#
#INPUT_2D_PATH='robotics_building_satellite.png'
#INPUT_2D_PATH='gothic_floorplans2.jpg'
#INPUT_2D_PATH='baroque_floorplans3.jpg'
#INPUT_2D_PATH='baroque_floorplans2.jpg'
#INPUT_2D_PATH='baroque_floorplans1.jpg'
#

#FILE_NAME_OBJ=150515_ComponentsA_21K
#FILE_NAME_OBJ=150515_ComponentsB_17K
#FILE_NAME_OBJ=150515_ComponentsC_21K
#OBJ_DIR=190515_components
#INPUT_OBJ_PATH=${OBJ_DIR}/${FILE_NAME_OBJ}.obj
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_A.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_B.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_C.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_D.obj'
#
#FILE_NAME_OBJ=190730_Rock_A
#FILE_NAME_OBJ=190730_Rock_B
#FILE_NAME_OBJ=190730_Rock_C
#FILE_NAME_OBJ=190730_Rock_D
FILE_NAME_OBJ=190730_Rock_E
INPUT_OBJ_PATH=${FILE_NAME_OBJ}.obj
#
#INPUT_2D=perspective2
#INPUT_2D=2
#INPUT_2D=presentationfield1
#INPUT_2D=presentationfield2
#INPUT_2D=softaccreations2
#INPUT_2D=sandra45
#INPUT_2D_PATH=matias_patterns/${INPUT_2D}.jpg
#
#INPUT_2D=new000001.png
INPUT_2D=new000101
#INPUT_2D=new000201.png
#INPUT_2D=new000301.png
#INPUT_2D=new000401.png
#INPUT_2D=new000501.png
#INPUT_2D=new000601.png
#INPUT_2D=new000701.png
#INPUT_2D=new000801.png
INPUT_2D_PATH=Tomography_05_HighRes/${INPUT_2D}.png
#
OUTPUT_FILENAME=2D${INPUT_2D}_3D_${FILE_NAME_OBJ}
OUTPUT_DIR=TEST
DIR_LOC=/mnt/ngv/askc-home/ARCHITECTURE_ROBOTICSGARDEN_PROJECT/Robotics_Building_networks
nvidia-docker run --rm -it \
	-v ${DIR_LOC}/3D_models_newboulders/:/3Dmodels:ro \
	-v ${DIR_LOC}/2Dmodels:/2Dmodels:ro \
	-v `pwd`:/root \
	2d3d-dreamstyle \
python run_examples_neural_renderer/example2.py \
    --filename_obj /3Dmodels/${INPUT_OBJ_PATH} \
    --filename_ref /2Dmodels/${INPUT_2D_PATH} \
    --filename_output_result /root/${OUTPUT_DIR}/${OUTPUT_FILENAME}.gif \
    --filename_output_optimization /root/${OUTPUT_DIR}/${OUTPUT_FILENAME}_optim.gif \
    --gpu 2
