# Welcome to the code repository for ARCH 660 Fall 2019, the Deep Design Studio
On this page you can find code that will allow you to explore different 2D-to-2D and 2D-to-3D image editing techniques, as well as collect simple datasets. 


It includes the following folders and files:

+ 2D_3D_style_dream_neural_renderer:
This folder contains code that allows you to perform style transfer, deep dreaming, or vertex optimization on mesh objects. Note that for style transfer and vertex optimization, a 2D guide image is required. More information on how to run the code is provided below. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE (see the tutorial provided below).

+ 2D_class_based_dreaming:
This folder contains code that allows you to perform the deep dreaming technique from Google, but allows you to specify an output class (such as fountain, arch, etc), meaning that you can hallucinate class features in images instead of arbitrary features learned by higher layer neurons. The folder also contains code that allows you to train your own classification neural network on a given dataset, which means you can specificy what image resolution you would like as well as what classes you would like. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE (see the tutorial provided below)

+ 2D_to_2D_neural_style_transfer:
This folder contains code that allows you to perform the 2D neural style transfer technique (used in Google's deep style editing GUI). However, with this code you can specify the input/output resolution. Note that the input would be a 2D 'content' image (i.e., the image whose spatial structure you wish to preserve) and a 2D 'style guide' image (i.e., an image whose style you wish to transfer onto the content image). NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE.

+ coding_tutorial_prt1:
This folder contains code that introduces you to basic python concepts (for loop, if statement, etc) as well as basic image processing in python. It also includes the powerpoint that will be given in class. 

+ paperspace-code-examples:
This folder provides two other deep learning examples that are designed to run on the paperspace cloud computing service: fast-style-transfer (which is a 'quick' version of the 2D to 2D neural style transfer above) and training/deploying pix2pix, which is a neural network architecure that performs domain transfer, ie, the network learns to transfer images from one dataset into the style of a separate dataset using generative adversarial network (GAN). To learn more about each project, tutorials are provided in the specific example project folders. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE (see the tutorial provided below)

+ simple_scrape_googleimages:
This folder contains code that allows you to generate a classification dataset by scraping google images. It requires a textfile of desired classes, which it uses as search words to google images. 

+ satelliteMapGeneration:
This is a code repository that allows you to download aerial/satellite maps given a latitude and logitude. This code is a bit tricky to use, and requires an account with maptiles. If you are interested in collecting a satellite dataset, please contact Sandra or Matias, and we can step you through the code. 

+ SuggestedReading.txt:
This file contains a list of resources about deep learning and architecture. 

The information provided below is a basic introduction into using Paperspace to run the different image editing techniques and paperspace examples. 

## THE BASICS OF PAPERSPACE:
A Gradient Project is a collection of experiments, code, metrics, and artifacts. Projects can be created manually or automatically from a Job corresponding to the current working directory name.
Experiments are used to create and start either a single Job or multiple Jobs (eg for a hyperparameter search or distributed training).  
Gradient Jobs are designed for executing code (such as training a deep neural network) on a CPU or GPU without managing any infrastructure.
Jobs are part of a larger suite of tools that work seamlessly with Gradient Notebooks, and our Core product, which together form a production-ready ML/AI pipeline.

A Job consists of:

- a collection of files (code, resources, etc.) from your local computer or GitHub
- a container (with code dependencies and packages pre-installed)
- a command to execute (i.e. python main.py or nvidia-smi)

## INSTALLING/SETTING UP PAPERSPACE:

The following describes the installation process for the paperspace CLI (command line interface) using paperspace-node. 
More detailed installation instructions can be found at: (https://www.npmjs.com/package/paperspace-node)

**Step 1:** Install npm and node.js on your local machine/computer
This is a package manager that we will use to install paperspace-node.
For detailed windows installation steps: (https://www.guru99.com/download-install-node-js.html) 

**Step 2:** Make a Paperspace account 
This is done via the website (https://www.paperspace.com). You need a credit card on file to use any machine.
NOTE: Sign up for gradient1 to access the paperspace computing machines (otherwise you are restricted to using machines, e.g., TK-80 that are hosted on google cloud engine, which means that you can only access data within your paperspace workspace and not in the paperspace /storage)

**Step 3:** Install Paperspace-node via npm locally on your computer
Paperspace-node is a way you can use the paperspace command line interface locally.
`npm install -g paperspace-node`
Note that the `-g` flag installs paperspace-node globally on your computer, which means you can access it from any directory on your filesystem.

**Step 4:** Store your paperspace API Key as an environment variable to make login easier
`export PAPERSPACE_API_KEY`
By setting the apiKey as an environment variable, you do not have to use the `--apiKey` flag during login. 

**Step 5:** Login to Paperspace from the command line on your local machine/computer
`paperspace login --apiKey {INSERT API KEY FROM WEBSITE HERE}`

Note that you can log out of paperspace using
`paperspace logout`


## RUNNING A JOB THROUGH PAPERSPACE CLI:

NOTE: for all jobs, the results of the network are saved to the /artifact folder in the paperspace workspace. 
This means that any input paths used in your scripts need to have /artifacts as their parent directory. /artifacts exist temporarily, and should be retrieved right after the job has completed. Alternatively, you can save results to 
/storage which persists across jobs and can be accessed by all notebooks/experiments.

### **2D Neural style transfer**

First, need to create notebook via web GUI, upload vgg weights to /storage

In terminal on local computer:

`cd 2D_to_2D_neural_style_transfer`

To initialize paperspace project, in terminal:

`paperspace project init`

To run job for this project, in terminal:

`paperspace jobs create --container acarlson32/neuralstyle-tf:firstimage --machineType P5000 --command "/paperspace/run_neural_style.sh" —-ignoreFiles "imagenet-vgg-verydeep-19.mat"`

To get the output style image from job (i.e., to  copy output files/folders from paperspace into your local directory), in terminal:

`paperspace jobs artifactsGet --jobID ps4r8vaor`

Note that `ps4r8vaor` is a jobID that is unique to the job you just ran. You need to get jobID from web gui job page.

A quick way to get the jobID of the most recent project/experiment is by running: 

`paperspace project show`

and copy the listed jobID

### **2D class-based deep dreaming**

When using a pertained network, you first need to upload the custom_weights folder to /storage using the notebook tool in the web GUI.

A few things to keep in mind:

	— NOTE THAT INPUT MUST BE IN RGB FORMAT (i.e., three channels)

	— You also have the option of uploading your dreamed images to the /storage folder, you would just need to specify their location in the appropriate run.sh file

In terminal on local computer:

`cd 2D_class_based_dreaming`

To initialize paperspace project, in terminal:

`paperspace project init`

To run a job for this project, in terminal:

`paperspace jobs create --container acarlson32/visclass-tf:firstimage --machineType P5000 --command "/paperspace/run_eval_paperspace.sh" --ignoreFiles "custom_weights"`

To get the output style image from job, in terminal:

`paperspace jobs artifactsGet --jobID ps4r8vaor`

To check the artifacts generated in the paperspace workspace:

`paperspace jobs artifactsList --jobId "j123abc" --size true`

To train a vgg16 network, follow the same steps but use the command:

`paperspace jobs create --container acarlson32/visclass-tf:firstimage --machineType P5000 --command "/paperspace/run_train_paperspace.sh" --ignoreFiles “custom_weights”`

### **2D to 3D Neural style transfer, 2D to 3D vertex optimization, and 3D deep dreaming**

This project uses the neural 3D mesh renderer (CVPR 2018) by H. Kato, Y. Ushiku, and T. Harada to achieve dreaming and style transfer in 3D. 
It builds upon the code in (https://github.com/hiroharu-kato/neural_renderer.git)

Note that before running any jobs in this project, you will need to upload the desired 3D models to the paperspace `/storage` space. Add each 3D model to `/storage/3Dmodels` and any 2D models (i.e., images) to `/storage/2Dmodels`.
You will also need to modify the appropriate bash file with the path locations (and other script parameters) before creating your job.

In terminal on local computer:

`cd 2D_3D_style_dream_neural_renderer`

To initialize paperspace project, in terminal:

`paperspace project init`

**To run a job for 2D to 3D style transfer, in terminal:**

`paperspace jobs create --container acarlson32/2d3d_neuralrenderer:firstimage --machineType P5000 --command "/paperspace/run_2d_to_3d_styletransfer.sh" —-ignoreFiles "results"`

**To run a job for 2D to 3D vertex optimization, in terminal:**

`paperspace jobs create --container acarlson32/2d3d_neuralrenderer:firstimage --machineType P5000 --command "/paperspace/run_2d_to_3d_vertexoptimization.sh" —-ignoreFiles "results"`

**To run a job for 3D deep dreaming, in terminal:**

`paperspace jobs create --container acarlson32/2d3d_neuralrenderer:firstimage --machineType P5000 --command "/paperspace/run_3d_deepdream.sh" —-ignoreFiles "results"`

To get the output style image from job (i.e., to  copy output files/folders from paperspace into your local directory), in terminal:

`paperspace jobs artifactsGet --jobID ps4r8vaor`

Note that `ps4r8vaor` is a jobID that is unique to the job you just ran. You need to get jobID from web gui job page.

A quick way to get the jobID of the most recent project/experiment is by running: 

`paperspace project show`

and copy the listed jobID
