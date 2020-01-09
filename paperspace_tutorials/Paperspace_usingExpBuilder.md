## Executing Models on Paperspace using Gradient Experiment Builder

## git repo
https://github.com/alexacarlson/DeepDesign2019.git

## Running 2D deep dream

# docker image
acarlson32/visclass-tf:firstimage

# Command Format
# bash run_2Ddeepdream_eval.sh IMAGE_DATA WEIGHTS_DIR DREAM_CLASS RESULTS_DIR NUM_ITERS 
bash run_2Ddeepdream_eval.sh /storage/2Dmodels/scene0_camloc_0_5_-20_rgb.png /storage/acadia_general_arch_styles_netweights gothic /storage/test 500 

## Running 2D style transfer

# docker image
acarlson32/neuralstyle-tf:firstimage

# Command Format
bash run_2Dneuralstyletransfer.sh /storage/2Dmodels/robotics_building_satellite.png /storage/2Dmodels/new000343.png /artifacts/roboticsbuilding_satellite_style000343_styleweight10.jpg 500 5.0 1.0

## Running 2D to 3D stuff

# docker image
acarlson32/2d3d_neuralrenderer:firstimage

# Command Format

bash run_2Dto3Ddeepdream.sh /storage/3Dmodels/bench.obj 3Ddreamed_bench.gif /artifacts/results_3D_dream 300

bash run_2Dto3Dstyletransfer.sh /storage/3Dmodels/TreeCartoon1_OBJ.obj /storage/2Dmodels/new000524.png 2Dgeo_3Dtree.gif /artifacts/results_2D_to_3D_styletransfer
