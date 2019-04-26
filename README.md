# RoboticsBuilding2019
The code repository for the garden design of the umich Robotics Building


## THE BASICS OF PAPERSPACE:
A Gradient Project is a collection of experiments, code, metrics, and artifacts. Projects can be created manually or automatically from a Job corresponding to the current working directory name.
Experiments are used to create and start either a single Job or multiple Jobs (eg for a hyperparameter search or distributed training).  
Gradient Jobs are designed for executing code (such as training a deep neural network) on a CPU or GPU without managing any infrastructure.
Jobs are part of a larger suite of tools that work seamlessly with Gradient Notebooks, and our Core product, which together form a production-ready ML/AI pipeline.
A Job consists of:
	— a collection of files (code, resources, etc.) from your local computer or GitHub
	— a container (with code dependencies and packages pre-installed)
	— a command to execute (i.e. python main.py or nvidia-smi)

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
By setting the apiKey as an environment variable, you do not have to use teh `--apiKey` flag during login 

**Step 5:** Login to Paperspace from the command line on your local machine/computer
`paperspace login --apiKey`

Note that you can log out of paperspace using
`paperspace logout`


## RUNNING A JOB THROUGH PAPERSPACE CLI:

NOTE: for all jobs, the results of the network are saved to the /artifact folder in the paperspace workspace. 
This means that any input paths used in your scripts need to have /artifacts as their parent directory. /artifacts exist temporarily, and should be retrieved right after the job has completed. Alternatively, you can save results to 
/storage which persists across jobs and can be accessed by all notebooks/experiments.

(1) 2D Neural style transfer
First, need to create notebook via web GUI, upload vgg weights to /storage

In terminal on local computer
`cd 2D_to_2D_neural_style_transfer_paperspace`

To initialize paperspace project, in terminal:
`paperspace project init`

To run job for this project, in terminal:
`paperspace jobs create --container acarlson32/neuralstyle-tf:firstimage --machineType P5000 --command "/paperspace/run_paperspace.sh" —-ignoreFiles “imagenet-vgg-verydeep-19.mat”`

To get the output style image from job (i.e., to  copy output files/folders from paperspace into your local directory), in terminal:
`paperspace jobs artifactsGet --jobID ps4r8vaor`

Note that `ps4r8vaor` is a jobID that is unique to the job you just ran. You need to get jobID from web gui job page.
A quick way to get the jobID of the most recent project/experiment is by running: 
`paperspace project show`
and copy the listed jobID

(2) 2D class-based deep dreaming

When using a pertained network, you first need to upload the custom_weights folder to /storage using the notebook tool in the web GUI.
A few things to keep in mind:
	— NOTE THAT INPUT MUST BE IN RGB FORMAT (i.e., three channels)
	— You also have the option of uploading your dreamed images to the /storage folder, you would just need to specify their location in the appropriate run.sh file

In terminal on local computer
`cd 2D_class_based_dreaming_paperspace`

To initialize paperspace project, in terminal:
`paperspace project init`

To run job for this project, in terminal:
`paperspace jobs create --container acarlson32/visclass-tf:firstimage --machineType P5000 --command "/paperspace/run_eval_paperspace.sh" --ignoreFiles “custom_weights”`

To get the output style image from job, in terminal:
`paperspace jobs artifactsGet --jobID ps4r8vaor`

To check the artifacts generated in the paperspace workspace:
`paperspace jobs artifactsList --jobId "j123abc" --size true`

To train a vgg16 network, follow the same steps but use the command:
`paperspace jobs create --container acarlson32/visclass-tf:firstimage --machineType P5000 --command "/paperspace/run_train_paperspace.sh" --ignoreFiles “custom_weights”`

