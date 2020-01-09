#INPUT_OBJ_PATH='/storage/3Dmodels/TreeCartoon1_OBJ.obj'
#INPUT_2D_PATH='/storage/2Dmodels/new000524.png'
#OUTPUT_FILENAME='2Dgeo_3Dtree'
#OUTPUT_DIR='/artifacts/results_vertoptim'
#NUM_ITERS=250

INPUT_OBJ_PATH=$1
INPUT_2D_PATH=$2
OUTPUT_FILENAME=$3
OUTPUT_DIR=$4
NUM_ITERS=$5

cd 2D_3D_style_dream_neural_renderer

python run_examples_neural_renderer/example2.py \
    --filename_obj ${INPUT_OBJ_PATH} \
    --filename_ref ${INPUT_2D_PATH} \
    --filename_output_optimization ${OUTPUT_DIR}/${OUTPUT_FILENAME}_optim.gif \
    --filename_output_result ${OUTPUT_DIR}/${OUTPUT_FILENAME}.gif \
    --num_iters ${NUM_ITERS}
