#INPUT_OBJ_PATH='190428_test_stairs_B.obj'
#INPUT_OBJ_PATH='Alexa/190502_Dreamground_D.obj'
INPUT_OBJ_PATH='cube/cube.obj'
OUTPUT_FILENAME='cube_recon.gif'
OUTPUT_DIR='results/results_3d_meshrecon'

nvidia-docker run --rm -it \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/:/3Dmodels:ro \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/2Dmodels:/2Dmodels:ro \
	-v `pwd`:/root \
	2d3d-dreamstyle \
python run_examples_neural_renderer/example1.py \
    --filename_input /3Dmodels/${INPUT_OBJ_PATH} \
    --filename_output /root/${OUTPUT_DIR}/${OUTPUT_FILENAME} \
    --gpu 2