INPUT_OBJ_PATH='/storage/3Dmodels/TreeCartoon1_OBJ.obj'
INPUT_2D_PATH='/storage/2Dmodels/new000524.png'
OUTPUT_FILENAME='2Dgeo_3Dtree.gif'
OUTPUT_DIR='/artifacts/results_2D_to_3D_styletransfer'
NUM_ITERS=10
STYLE_W = 1.0
STYLE_C = 2e9
 
python ./run_examples_style_transfer_3d/run.py \
    --filename_mesh ${INPUT_OBJ_PATH} \
    --filename_style ${INPUT_2D_PATH} \
    --filename_output ${OUTPUT_DIR}/${OUTPUT_FILENAME}
    --num_iteration ${NUM_ITERS} \
    --lambda_style ${STYLE_W} \
    --lambda_content ${STYLE_C}

##FILE_NAME_OBJ=150515_ComponentsA_21K
##FILE_NAME_OBJ=150515_ComponentsB_17K
##FILE_NAME_OBJ=150515_ComponentsC_21K
##OBJ_DIR=190515_components
##INPUT_OBJ_PATH=${OBJ_DIR}/${FILE_NAME_OBJ}.obj
##INPUT_OBJ_PATH='Alexa/190502_Dreamground_A.obj'
##INPUT_OBJ_PATH='Alexa/190502_Dreamground_B.obj'
##INPUT_OBJ_PATH='Alexa/190502_Dreamground_C.obj'
##INPUT_OBJ_PATH='Alexa/190502_Dreamground_D.obj'#

##FILE_NAME_OBJ=190730_Rock_A
##FILE_NAME_OBJ=190730_Rock_B
##FILE_NAME_OBJ=190730_Rock_C
##FILE_NAME_OBJ=190730_Rock_D
##FILE_NAME_OBJ=190730_Rock_E
#FILE_NAME_OBJ=Rock_A_Res_2
##FILE_NAME_OBJ=Rock_A_Res_3
##FILE_NAME_OBJ=Rock_A_Res_4
#INPUT_OBJ_PATH=${FILE_NAME_OBJ}.obj
##
##INPUT_2D=perspective2
##INPUT_2D=2
##INPUT_2D=presentationfield1
##INPUT_2D=presentationfield2
##INPUT_2D=softaccreations2
##INPUT_2D=sandra45
##
##INPUT_2D=new000001.png
##INPUT_2D=new000101.png
##INPUT_2D=new000201.png
##INPUT_2D=new000301.png
##INPUT_2D=new000401.png
#INPUT_2D=new000501.png
##INPUT_2D=new000601.png
##INPUT_2D=new000701.png
##INPUT_2D=new000801.png
#INPUT_2D_PATH=Tomography_05_HighRes/${INPUT_2D}
##
##INPUT_2D_PATH='new000343.png'
##INPUT_2D_PATH='new000524.png'
##
#OUTPUT_FILENAME=2D${INPUT_2D}_${FILE_NAME_OBJ}.gif
##OUTPUT_FILENAME='2Dnew000343_3D_robotics_stairs_E'
#OUTPUT_DIR='results_2D_to_3D_styletransfer_robotgarden2'
##
#IMAGE_SIZE=400#

#nvidia-docker run --rm -it \
#	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/3D_Boulders_Resolution/:/3Dmodels:ro \
#	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/2Dmodels:/2Dmodels:ro \
#	-v `pwd`:/root \
#	2d3d-dreamstyle \
#python run_examples_style_transfer_3d/run.py \
#    --filename_mesh /3Dmodels/${INPUT_OBJ_PATH} \
#    --filename_style /2Dmodels/${INPUT_2D_PATH} \
#    --filename_output /root/${OUTPUT_DIR}/${OUTPUT_FILENAME} \
#    --image_size ${IMAGE_SIZE} \
#    --camera_distance 2.732 \
#    --elevation_max 1000.0 \
#    --elevation_min 0.
#    --gpu 1
