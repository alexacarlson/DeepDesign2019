
#INPUT_OBJ_PATH='/storage/3Dmodels/bench.obj'
#OUTPUT_FILENAME=3Ddreamed_bench.gif
#OUTPUT_DIR=/artifacts/results_3D_dream
#NUM_ITER=300

INPUT_OBJ_PATH=$1
OUTPUT_FILENAME=$2
OUTPUT_DIR=$3
IMAGE_SIZE=$4
NUM_ITER=$5

#cd 2D_3D_style_dream_neural_renderer
cd 2d_3d_style_dream_neural_renderer

python run_examples_deep_dream_3d/run.py \
    --filename_obj ${INPUT_OBJ_PATH} \
    --output_directory ${OUTPUT_DIR} \
    --filename_output ${OUTPUT_DIR}/${OUTPUT_FILENAME} \
    --image_size ${IMAGE_SIZE} \
    --num_iteration ${NUM_ITER}
