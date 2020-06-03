# Welcome to the code repository for ARCH 660/662, the Deep Design Studio
On this page you can find code that will allow you to explore different 2D-to-2D and 2D-to-3D image editing techniques, as well as collect simple datasets. Note that Paperspace is a cloud computing platform that allows you to access GPUs, which are necessary to run deep learning models efficiently. A tutorial on how to use the code in this repository with Paperspace is given in the following section.

### Quickstart
For step by step instructions on using Paperspace, please see the following:

[The Basics of Paperspace](paperspace_tutorials/Paperspace_basics.md)

[Uploading your dataset to and downloading your model outputs from Paperspace](paperspace_tutorials/Paperspace_uploadingdata.md)

[Running and training models using the Gradient Experiment Builder](paperspace_tutorials/Paperspace_usingExpBuilder.md)

[Model Zoo](paperspace_tutorials/model_zoo.md)

[Running/training models using the Paperspace Command Line Interface (NOT RECOMMENDED)](paperspace_tutorials/Paperspace_usingtheCLI.md)



It includes the following folders and files:

+ 2D_3D_style_dream_neural_renderer:
This folder contains code that allows you to perform style transfer, deep dreaming, or vertex optimization on mesh objects. Note that for style transfer and vertex optimization, a 2D guide image is required. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE. See the tutorials provided below.

+ 2D_class_based_dreaming:
This folder contains code that allows you to perform the deep dreaming technique from Google, but allows you to specify an output class (such as fountain, arch, etc), meaning that you can hallucinate class features in images instead of arbitrary features learned by higher layer neurons. The folder also contains code that allows you to train your own classification neural network on a given dataset, which means you can specificy what image resolution you would like as well as what classes you would like. NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE. See the tutorials provided below.

+ 2D_to_2D_neural_style_transfer:
This folder contains code that allows you to perform the 2D neural style transfer technique (used in Google's deep style editing GUI). However, with this code you can specify the input/output resolution. Note that the input would be a 2D 'content' image (i.e., the image whose spatial structure you wish to preserve) and a 2D 'style guide' image (i.e., an image whose style you wish to transfer onto the content image). NOTE YOU WILL NEED TO USE PAPERSPACE TO RUN THIS CODE. See the tutorials provided below.

+ google_DeepDreaming:
This folder contains code that underlies the google Deep Dream web UI. Thus, the key differences between the above class-based dreaming and this function is that, with this code, you can hallucinate learned features of *neurons* at *multiple scales* in input images. Also note that this code can take in any resolution input image, which is restricted in the web UI.

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
  
  + deeplearning_architecture_semseries_updated.pptx:
  This is a powerpoint presentation that contains basic concepts and mathematics behind deep learning.

+ unzipfile.sh: this file can be used to unzip `.zip` files that you have uploaded into the Paperspace Persistent Storage location. See the tutorial on uploading and downloading data on Paperspace listed below for more detail on usage.

+ zipfile.sh: this file can be used to zip folders that are in Paperspace Persistent Storage location. See the tutorial on uploading and downloading data on Paperspace listed below for more detail on usage.

The information provided below is a basic introduction into using Paperspace to run the different image editing techniques and paperspace examples. 



