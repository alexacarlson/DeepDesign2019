#INPUT_OBJ_PATH='/storage/3Dmodels/TreeCartoon1_OBJ.obj'
#INPUT_2D_PATH='/storage/2Dmodels/new000524.png'
#OUTPUT_FILENAME='2Dgeo_3Dtree.gif'
#OUTPUT_DIR='/artifacts/results_2D_to_3D_styletransfer'
#STYLE_WEIGHT=1.0
#CONTENT_WEIGHT=2e9
#NUM_ITERS=1000

INPUT_OBJ_PATH=$1
INPUT_2D_PATH=$2
OUTPUT_FILENAME=$3
OUTPUT_DIR=$4
STYLE_WEIGHT=$5
CONTENT_WEIGHT=$6
NUM_ITERS=$7

cd 2D_3D_style_dream_neural_renderer/

python run_examples_style_transfer_3d/run.py \
    --filename_mesh ${INPUT_OBJ_PATH} \
    --filename_style ${INPUT_2D_PATH} \
    --filename_output ${OUTPUT_DIR}/${OUTPUT_FILENAME} \
    --lambda_style ${STYLE_WEIGHT} \
    --lambda_content ${CONTENT_WEIGHT} \
    --num_iteration ${NUM_ITERS}
