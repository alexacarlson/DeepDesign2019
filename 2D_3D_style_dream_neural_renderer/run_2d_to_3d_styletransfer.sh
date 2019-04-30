INPUT_OBJ_PATH='/storage/3Dmodels/TreeCartoon1_OBJ.obj'
INPUT_2D_PATH='/storage/2Dmodels/tree1_res.jpg'
OUTPUT_FILENAME='2Dtree_3Dstairs2.gif'
OUTPUT_DIR='/artifacts/results_2D_to_3D_styletransfer'
 
python run_examples_style_transfer_3d/run.py \
    --filename_mesh ${INPUT_OBJ_PATH} \
    --filename_style ${INPUT_2D_PATH} \
    --filename_output ${OUTPUT_DIR}/${OUTPUT_FILENAME}
