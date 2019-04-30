INPUT_OBJ_PATH='/storage/3Dmodels/190428_test_stairs_E.obj'
OUTPUT_FILENAME=3Ddreamed_stairs.gif
OUTPUT_DIR=/artifacts/results_3D_dream
NUM_ITER=300

python run_examples_deep_dream_3d/run.py \
    --filename_obj ${INPUT_OBJ_PATH} \
    --output_directory ${OUTPUT_DIR} \
    --filename_output ${OUTPUT_DIR}/${OUTPUT_FILENAME} \
    --num_iteration ${NUM_ITER}
