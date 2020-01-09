# Executing Models using Paperspace Gradient Experiment Builder
The Paperspace Experiment Builder is a wizard-style UI tool to submit a job. You can use it for both training and testing models.  

After signing in, click on Gradient in the navigation bar on the left to choose Projects. Click on the Create Project button and select Create Standalone Project. Enter a name for the project when prompted.

Once you have created a Project, select the project to enter its console.

Select the light blue `+ Create Experiment` button on the right hand side of the Project console.
This takes you through various options to set up your Experiment.

The first step is to choose a machine type for scheduling the Experiment.

In Gradient, Experiments are based on a container image that provides the runtime and dependencies.

With the runtime container in place, we now need to point Gradient to the dataset and the training script. This is done through the integration with GitHub. Gradient pulls the GitHub repo into the experiment workspace and uses the assets for the training job.

Finally, let’s define the command for the job which is the Python script that executes within the context of the runtime of the container. When the script exits gracefully, the job is marked as complete.

We are now ready to kick off the training job by clicking on the Submit Experiment button.
Gradient adds the job to the queue and schedules it in one of the chosen machine types. In a few minutes, the job execution completes.

## git repo
https://github.com/alexacarlson/DeepDesign2019.git

## Running 2D deep dream

2D deep dream Docker image:

`acarlson32/visclass-tf:firstimage`

Command Format:
`bash run_2Ddeepdream_eval.sh IMAGE_DATA WEIGHTS_DIR DREAM_CLASS RESULTS_DIR NUM_ITERS`

Command Example:

`bash run_2Ddeepdream_eval.sh /storage/2Dmodels/scene0_camloc_0_5_-20_rgb.png /storage/acadia_general_arch_styles_netweights gothic /storage/test 500`

## Running 2D style transfer

2D style transfer docker image:

`acarlson32/neuralstyle-tf:firstimage`

Command Format:

`bash run_2Dneuralstyletransfer.sh CONTENT_FILE STYLE_FILE OUTPUT_FILE IMAGE_SIZE CONTENT_WEIGHT STYLE_WEIGHT`

Command Example: 

`bash run_2Dneuralstyletransfer.sh /storage/2Dmodels/robotics_building_satellite.png /storage/2Dmodels/new000343.png /artifacts/roboticsbuilding_satellite_style000343_styleweight10.jpg 500 5.0 1.0`

## Running 2D to 3D neural renderer

Neural Renderer docker image:

`acarlson32/2d3d_neuralrenderer:firstimage`

Command Format:

bash run_2Dto3Ddeepdream.sh /storage/3Dmodels/bench.obj 3Ddreamed_bench.gif /artifacts/results_3D_dream 300

bash run_2Dto3Dstyletransfer.sh /storage/3Dmodels/TreeCartoon1_OBJ.obj /storage/2Dmodels/new000524.png 2Dgeo_3Dtree.gif /artifacts/results_2D_to_3D_styletransfer

bash run_2Dto3Dvertexoptimization.sh /storage/3Dmodels/TreeCartoon1_OBJ.obj /storage/2Dmodels/new000524.png 2Dgeo_3Dtree /artifacts/results_vertoptim 250
