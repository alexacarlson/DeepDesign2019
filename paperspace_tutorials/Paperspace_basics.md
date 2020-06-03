
## THE BASICS OF PAPERSPACE:
Paperspace is a cloud computing service that gives you access to powerful machines for running/training models. Gradient is a project management platform provided by Paperspace. We can use it to easily deploy deep learning models. 
The folders listed in the previous section can each be considered a Gradient Project. 
A Gradient Project is a workspace for you to run Experiments and Jobs, store Artifacts (such as models and code), and manage Deployments (deployed models). We will be creating StandAlone Projects exclusively (vs. using the GradientCI project framework). 

Experiments are used to train machine learning models. When you run an Experiment within Gradient, a job is created and executed. Jobs are designed for executing code (such as training a deep neural network) on a CPU or GPU without managing any infrastructure. You can run multiple jobs within a single experiment.

A Job consists of a collection of files (code, resources, etc.) from your local computer or GitHub, a container (with code dependencies and packages pre-installed), and a command to execute (i.e. python main.py). It is recommended to use the Gradient Experiment builder to run jobs. A step-by-step tutorial on how to use this feature is given below.

Paperspace also has several ways in which you can store your code, dataset, and model outputs (e.g., network weights, output images, accuracy metrics, etc). The first, called Persistent storage, is a persistent filesystem automatically mounted on every Experiment, Job, and Notebook and is ideal for storing data like images, datasets, model checkpoints, and more. Anything you store in `/storage` directory will be persistently stored in a given storage region (e.g., East vs. west coast storage).

The second, called Workspace storage, is typically imported from the local directory in which you started your job. The contents of that directory are zipped up and uploaded to the container in which your job runs. The Workspace exists for the duration of the job run.  

The third, called Artifact storage, is collected and made available after the Experiment or Job run is completed. You can download any files that your job has placed in the /artifacts directory from the web UI. If you need to get result data, such as images, from a job run out of Gradient, use the Artifacts directory. Note that the total of Artifact storage cannot exceed ~200 GB. If you find you need more space, then it is recommended to use Persistent storage.
