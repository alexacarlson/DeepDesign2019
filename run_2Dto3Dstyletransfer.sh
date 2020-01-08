INPUT_OBJ_PATH='/storage/3Dmodels/TreeCartoon1_OBJ.obj'
INPUT_2D_PATH='/storage/2Dmodels/new000524.png'
OUTPUT_FILENAME='2Dgeo_3Dtree.gif'
OUTPUT_DIR='/artifacts/results_2D_to_3D_styletransfer'
 
cd 2D_3D_style_dream_neural_renderer/

python run_examples_style_transfer_3d/run.py \
    --filename_mesh ${INPUT_OBJ_PATH} \
    --filename_style ${INPUT_2D_PATH} \
    --filename_output ${OUTPUT_DIR}/${OUTPUT_FILENAME}
