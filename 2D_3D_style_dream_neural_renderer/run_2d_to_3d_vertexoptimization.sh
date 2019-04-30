INPUT_OBJ_PATH='/storage/3Dmodels/TreeCartoon1_OBJ.obj'
INPUT_2D_PATH='/storage/2Dmodels/tree1.jpg'
OUTPUT_FILENAME='2Dgeo_3Dstairs'
OUTPUT_DIR='/artifacts/results_vertoptim'
NUM_ITERS=250

python run_examples_neural_renderer/example2.py \
    --filename_obj ${INPUT_OBJ_PATH} \
    --filename_ref ${INPUT_2D_PATH} \
    --filename_output_optimization ${OUTPUT_DIR}/${OUTPUT_FILENAME}_optim.gif \
    --filename_output_result ${OUTPUT_DIR}/${OUTPUT_FILENAME}.gif \
    --num_iters ${NUM_ITERS}
