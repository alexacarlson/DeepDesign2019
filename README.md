# Welcome to the code repository for ARCH 660, the Deep Design Studio
On this page you can find code that will allow you to explore different 2D-to-2D and 2D-to-3D image editing techniques, as well as collect simple datasets. Note that Paperspace is a cloud computing platform that allows you to access GPUs, which are necessary to run deep learning models efficiently. A tutorial on how to use the code in this repository with Paperspace is given in the following section.


It includes the following folders and files:

+ 2D_3D_style_dream_neural_renderer:
This folder contains code that allows you to perform style transfer, deep dreaming, or vertex optimization on mesh objects. Note that for style transfer and vertex optimization, a 2D guide image is required. More information on how to run the code is provided below. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE. [See the tutorial provided below](#running-and-training-models-using-the-gradient-experiment-builder).

+ 2D_class_based_dreaming:
This folder contains code that allows you to perform the deep dreaming technique from Google, but allows you to specify an output class (such as fountain, arch, etc), meaning that you can hallucinate class features in images instead of arbitrary features learned by higher layer neurons. The folder also contains code that allows you to train your own classification neural network on a given dataset, which means you can specificy what image resolution you would like as well as what classes you would like. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE (see the tutorial provided below).

+ 2D_to_2D_neural_style_transfer:
This folder contains code that allows you to perform the 2D neural style transfer technique (used in Google's deep style editing GUI). However, with this code you can specify the input/output resolution. Note that the input would be a 2D 'content' image (i.e., the image whose spatial structure you wish to preserve) and a 2D 'style guide' image (i.e., an image whose style you wish to transfer onto the content image). NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE (see the tutorial provided below).

+ dataset_generation: 
This folder contains code that can be used for dataset generation. Descriptions of its functionality are below:

  + simple_scrape_googleimages:
  This folder contains code that allows you to generate a classification dataset by scraping google images. It requires a     textfile of desired classes, which it uses as search words to google images. 

  + satelliteMapGeneration:
  This is a code repository that allows you to download aerial/satellite maps given a latitude and logitude. This code is a bit tricky to use, and requires an account with maptiles. If you are interested in collecting a satellite dataset, please contact Sandra or Matias, and we can step you through the code. 

+ class_material:
This folder contains material that we have discussed during lecture.
  + coding_tutorial_prt1:
  This folder contains code that introduces you to basic python concepts (for loop, if statement, etc) as well as basic image processing in python. It also includes the powerpoint that will be given in class. 

  + SuggestedReading.txt:
  This file contains a list of resources about deep learning and architecture. 

The information provided below is a basic introduction into using Paperspace to run the different image editing techniques and paperspace examples. 

## THE BASICS OF PAPERSPACE:
Gradient is a project management platform provided by Paperspace. We can use it to easily deploy deep learning models. 
The folders listed in the previous sections can be considered a Gradient Project. 
A Gradient Project is a workspace for you to run Experiments and Jobs, store Artifacts (such as models and code), and manage Deployments (deployed models). We will be creating StandAlone Projects exclusively (vs. using the GradientCI project framework). 

Experiments are used to train machine learning models. When you run an Experiment within Gradient, a job is created and executed. Jobs are designed for executing code (such as training a deep neural network) on a CPU or GPU without managing any infrastructure. You can run multiple jobs within a single experiment.

A Job consists of a collection of files (code, resources, etc.) from your local computer or GitHub, a container (with code dependencies and packages pre-installed), and a command to execute (i.e. python main.py). It is recommended to use the Gradient Experiment builder to run jobs. A step-by-step tutorial on how to use this feature is given below.

Paperspace also has several ways in which you can store your code, dataset, and model outputs (e.g., network weights, output images, accuracy metrics, etc). The first, called Persistent storage, is a persistent filesystem automatically mounted on every Experiment, Job, and Notebook and is ideal for storing data like images, datasets, model checkpoints, and more. Anything you store in `/storage` directory will be persistently stored in a given storage region.

The second, called Workspace storage, is typically imported from the local directory in which you started your job. The contents of that directory are zipped up and uploaded to the container in which your job runs. The Workspace exists for the duration of the job run.  

The third, called Artifact storage, is collected and made available after the Experiment or Job run in the CLI and web interface. You can download any files that your job has placed in the /artifacts directory from the CLI or UI. If you need to get result data from a job run out of Gradient, use the Artifacts directory. 

For step by step instructions on using Paperspace, please see the following:

[Uploading your dataset to and downloading your model outputs from Paperspace](paperspace_tutorials/Paperspace_uploadingdata.md)

[Running and training models using the Gradient Experiment Builder](paperspace_tutorials/Paperspace_usingExpBuilder.md)

[Running/training models using the Paperspace Command Line Interface (NOT RECOMMENDED)](paperspace_tutorials/Paperspace_usingtheCLI.md)

