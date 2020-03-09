# Model Zoo
Collection of more deep learning models, with tutorials on how to run them on paperspace.

## Content
1. [AttnGAN](#attention_gan)
2. [Google's Neuron-based Deep Dream](#googledeepdream)
3. [Class-Based Deep Dream](#classdeepdream)
4. [2D Neural Style Transfer](#neuralstyle)
5. [2D to 3D Deep Dreaming](#3ddream)
6. [2D to 3D Style Transfer](#3dstyle)
7. [2D to 3D Vertex Optimizatoin](#3dvert)
8. [Pix2PixHD](#pix2pixhd)
9. [Pix2Pix](#pix2pix)
10. [CycleGAN](#cyclegan)
11. [PG-GAN](#pggan)
12. [StyleGan2](#stylegan)

<a name="attention_gan"></a>
## AttnGAN for Image generation from Text 
Pytorch implementation for reproducing AttnGAN results in the paper [AttnGAN: Fine-Grained Text to Image Generation
with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He. (This work was performed when Tao was an intern with Microsoft Research). 
Currently, it uses the weights trained on COCO dataset.

### Running AttnGAN
+ Upload and unzip the [coco.zip](https://drive.google.com/file/d/1BEwvpCyLvuTFsFuWlmX1NfWS13v0HcY4/view?usp=sharing) file in the storage folder using Jupyter notebook
+ In the coco folder modify `captions.txt` file for the desired text/caption input
+ Start an experiment with the following parameters in paperspace:
	+ **Container:** `brannondorsey/docker-stackgan`
    + **Workspace:** `https://github.com/dysdsyd/AttnGAN.git`
    + **Command:** `bash run.sh`
+ Generated images will be dumped in the `storage/coco/coco_AttnGAN2/captions` directory for each input sentence in the `captions.txt` file. Each sentence generates 5 image files and the file with `*_g2.png` extension is the final output.


<a name="googledeepdream"></a>
## Google Deep Dream
DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image.
It does so by forwarding an image through the network, then calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations, enhancing the patterns seen by the network, and resulting in a dream-like image. This process was dubbed "Inceptionism". 
This is Google's implementation of deep dream that is used for their web UI. 

This code allows you to augment an input image with the learned features for a specifc neuron or every neuron in the final layer of a classification neural network. Thus, for a single input image, this code will either generate a single output image, or generate 143 output images, each using the learned features from a different neuron. 
When using a pre-trained network, you first need to upload the `inception5_weights.zip` file folder to `/storage` using the notebook tool in the web GUI, and unzip it. What is nice about using this code in leu of the google web UI is that this code takes in an image of any resolution!

2D deep dream Docker container image:

`tensorflow/tensorflow:1.1.0-devel-gpu`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format:

`bash run_2Dgoogledeepdream_eval.sh IMAGE_DATA WHICH_NEURON MODEL_DIR RESULTS_DIR NUM_ITERS`

Where `IMAGE_DATA` is the location of your input image on paperspace, `WHICH_NEURON` is a number 0-143 to specify which neuron you would like to use for dreaming. If you would like to use all of them, specify `'all'` instead of a number. `MODEL_DIR` is the location of `inception5_weights` in paperspace, `RESULTS_DIR` is the location in paperspace where the output image(s) will be saved, and `NUM_ITERS` determines how many times the algorithm will perform the dreaming operation (more iterations will have a stronger dreaming effect in the input image).

Command Example:

`bash run_2Dgoogledeepdream_eval.sh /storage/2Dmodels/tree1.jpg /storage/inception5h_weights/tensorflow_inception_graph.pb /storage/test_dream 30`

<a name="classdeepdream"></a>
## Class-based Deep Dream
This code allows you to augment an input image with the learned features of a specific class from a trained classification neural network, specifically a VGG network. 
This technique is used for visualising the class models learnt by the image classification ConvNets. Given a learnt/trained classification ConvNet and a class of interest from the network's training dataset, this visualisation method generates an image with features that are representative of what the ConvNet has learned to represent/detect the given class; it lets us know what are the features expected in input image to maximize the output class node score.

### Evaluating 2D class-based deep dream
When using a pre-trained network, you first need to upload pretrained VGG neural network weights folder to `/storage` using the notebook tool in the web GUI. Note that the next subsection details how to train a VGG network on your own dataset. 
Note that the  input must be in RGB format (i.e., three channels).

2D deep dream Docker container image:

`acarlson32/visclass-tf:firstimage`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format:

`bash run_2Dclassbaseddeepdream_eval.sh IMAGE_DATA WEIGHTS_DIR DREAM_CLASS RESULTS_DIR NUM_ITERS IMAGE_H IMAGE_W`

Where `IMAGE_DATA` is the location of your input image on paperspace, `WEIGHTS_DIR` is the location of the VGG network weights in paperspace,`DREAM_CLASS` is the class you would like to use for dreaming. If you would like to use all of them, specify `'all'` instead of a number.  `RESULTS_DIR` is the location in paperspace where the output image(s) will be saved, and `NUM_ITERS` determines how many times the algorithm will perform the dreaming operation (more iterations will have a stronger dreaming effect in the input image), `IMAGE_H` and `IMAGE_W` are the output image height and width, respectively.

Command Example:

`bash run_2Dclassbaseddeepdream_eval.sh /storage/2Dmodels/scene0_camloc_0_5_-20_rgb.png /storage/acadia_general_arch_styles_netweights gothic /storage/test 500 720 1280`

### Training a classifcation network on your dataset to use for 2D class-based deep dream
This code produces a trained VGG network that can be used to perform 2D class-based dreaming (as described in the previous section). Note that the weights directory in the below command is where the weights will be saved. You will need to upload your classification dataset to `/storage` before running this code. It assumes that your dataset contains a folder for each different class of images, and that each of these folders contains images only for that class. 

2D deep dream Docker container image:

`acarlson32/visclass-tf:firstimage`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format:

`bash run_2Dclassbaseddeepdream_training.sh TRAIN_IMAGE_DIR TRAIN_EPOCHS WEIGHTS_DIR RESULTS_DIR`

Where `TRAIN_IMAGE_DIR` is the location of your classification training dataset on paperspace, `TRAIN_EPOCHS` determines how long your classification network will train for, `WEIGHTS_DIR` is the location of where the finetuned VGG network weights will be saved in paperspaced, , `IMAGE_H` and `IMAGE_W` are the desired image height and width that your trained VGG network will operate on, respectively. Note that the network can only operate on images that are the same size as it has been trained for.

Command Example:

`bash run_2Dclassbaseddeepdream_training.sh /storage/classificationdataset /storage/acadia_general_arch_styles_netweights gothic /storage/test 500`

<a name="neuralstyle"></a>
## Neural Style Transfer
Neural style transfer is an optimization technique used to take two images—a content image and a style reference image (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.
This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network.
You can run this code with no mask, which means that the style will be transferred to the entire content image, or with a mask, which will apply the style to only the locations in the content image specified by the mask. Currently, the mask is a binary mask the same size as the content image, where a pixel value of 1 indicates a region to transfer style to and 0 indicates where the content image will be unaffected by the style transfer. 

### Running 2D style transfer
First, you will need to create notebook via web GUI, upload vgg weights, called `imagenet-vgg-verydeep-19.mat`, to `/storage`
You will also need to upload a content image and style guide image, and if desired the mask image, to `/storage`. You can download the vgg weights from <http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat>.

2D style transfer Docker container image:

`acarlson32/neuralstyle-tf:firstimage`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format for running with no mask:

`bash run_2Dneuralstyletransfer_nomask.sh CONTENT_FILE STYLE_FILE OUTPUT_DIR IMAGE_SIZE CONTENT_WEIGHT STYLE_WEIGHT NUM_ITERS`

Command Example for running with no mask: 

`bash run_2Dneuralstyletransfer_nomask.sh /storage/2Dmodels/robotics_building_satellite.png /storage/2Dmodels/new000343.png /artifacts 500 5.0 1.0 100`


Command Format for running with mask:

`bash run_2Dneuralstyletransfer_withmask.sh CONTENT_FILE STYLE_FILE MASK_FILE OUTPUT_DIR IMAGE_SIZE CONTENT_WEIGHT STYLE_WEIGHT NUM_ITERS`

Command Example  for running with mask: 

`bash run_2Dneuralstyletransfer_withmask.sh /storage/2Dmodels/robotics_building_satellite.png /storage/2Dmodels/new000343.png /storage/2Dmodels/new000343_mask.png /artifacts/styleoutputdir 500 5.0 1.0 100`

<a name="3ddream"></a>
### Running 2D to 3D neural renderer for 3D deep dreaming
This project uses the neural 3D mesh renderer (CVPR 2018) by H. Kato, Y. Ushiku, and T. Harada to achieve dreaming and style transfer in 3D. It builds upon the code in (https://github.com/hiroharu-kato/neural_renderer.git)

Note that before running any jobs in this project, you will need to upload the desired 3D models (in `.obj` format) to the paperspace `/storage space`. Add each 3D model to `/storage/3Dmodels`. You do not need to worry about uploading pretrained weights, the code handles this under the hood. 

Neural Renderer Docker container image:

`acarlson32/2d3d_neuralrenderer:secondimage`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format:

`bash run_2Dto3Ddeepdream.sh INPUT_OBJ_PATH OUTPUT_FILENAME OUTPUT_DIR IMAGE_SIZE NUM_ITER`

Command Example: 

`bash run_2Dto3Ddeepdream.sh /storage/3Dmodels/bench.obj 3Ddreamed_bench.gif /artifacts/results_3D_dream 512 300`

<a name="3dstyle"></a>
### Running 2D to 3D neural renderer for 2D to 3D style transfer
This project uses the neural 3D mesh renderer (CVPR 2018) by H. Kato, Y. Ushiku, and T. Harada to achieve dreaming and style transfer in 3D. It builds upon the code in (https://github.com/hiroharu-kato/neural_renderer.git)

Note that before running any jobs in this project, you will need to upload the desired 3D models (in `.obj` format) to the paperspace `/storage` space. Add each 3D model to `/storage/3Dmodels` and any 2D models (i.e., images) to `/storage/2Dmodels`. You do not need to worry about uploading pretrained weights, the code handles this under the hood. 

Neural Renderer Docker container image:

`acarlson32/2d3d_neuralrenderer:secondimage`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format:

`bash run_2Dto3Dstyletransfer.sh INPUT_OBJ_PATH INPUT_2D_PATH OUTPUT_FILENAME OUTPUT_DIR STYLE_WEIGHT CONTENT_WEIGHT NUM_ITERS`

Command Example: 

`bash run_2Dto3Dstyletransfer.sh /storage/3Dmodels/TreeCartoon1_OBJ.obj /storage/2Dmodels/new000524.png 2Dgeo_3Dtree.gif /artifacts/results_2D_to_3D_styletransfer 1.0 2e9 1000`

<a name="3dvert"></a>
### Running 2D to 3D neural renderer for 2D to 3D vertex optimization
This project uses the neural 3D mesh renderer (CVPR 2018) by H. Kato, Y. Ushiku, and T. Harada to achieve dreaming and style transfer in 3D. It builds upon the code in (https://github.com/hiroharu-kato/neural_renderer.git)

Note that before running any jobs in this project, you will need to upload the desired 3D models (in `.obj` format) to the paperspace `/storage` space. Add each 3D model to `/storage/3Dmodels` and any 2D models (i.e., images) to `/storage/2Dmodels`. You do not need to worry about uploading pretrained weights, the code handles this under the hood. 

Neural Renderer Docker container image:

`acarlson32/2d3d_neuralrenderer:secondimage`

Workspace:

`https://github.com/alexacarlson/DeepDesign2019.git`

Command Format:

`bash run_2Dto3Dvertexoptimization.sh INPUT_OBJ_PATH INPUT_2D_PATH OUTPUT_FILENAME OUTPUT_DIR NUM_ITERS`

Command Example: 

`bash run_2Dto3Dvertexoptimization.sh /storage/3Dmodels/TreeCartoon1_OBJ.obj /storage/2Dmodels/new000524.png 2Dgeo_3Dtree /artifacts/results_vertoptim 250`

<a name="pix2pixhd"></a>
## Pix2pixHD for paired image-to-image translation
We will step through how to train and test the super resolution GAN model, `pix2pixHD` in the Paperspace Experiment Builder. 

First, `pix2pixHD` is a generative adversarial neural network that transforms one dataset of high resolution images, which we refer to as data domain A, into the style of a different high resolution dataset, which we refer to as data domain B. Note that the data in domain A must be paired with the data in domain B; this means that the spatial structure of an image in domain A must correpsond to an image in domain D that has the same spatial structure; a good example of this is having domain A be a collection of semantic segmentation maps (each object class is a different color pixel) and domain B is the segmentation maps corresponding RGB image. This means that, effectively, pix2pixHD is painting the RGB color palette/textures of the shapes in domain B onto the appropriate shapes in domain A.
We use the term domain to desrcibe a dataset because a limited amount of visual information is captured in each dataset, and can vary greatly between datasets. Thus, in a sense, each dataset is it's own little world, or domain! The easiest way to understand this is by thinking of a dataset of purely daytime images compared to a dataset of purely night time images. While both datasets may capture similar structures (buildings, roads cars, people etc), the overall appearance/style is drastically different. The `pix2pixHD` was originally developed to transform semantically segmented maps into corresponding images, but it can be trained to transfer any one dataset into the style of a different dataset. 

The pix2pixHD Docker container you can use for both training and testing your model:

`taesungp/pytorch-cyclegan-and-pix2pix`

The workspace you can use for both training and testing your model:

https://github.com/NVIDIA/pix2pixHD.git

### Training pix2pixHD
For training pix2pixHD, you will need to upload your input data domain A to `/storage/train_A`, and your output data domain B to `/storage/train_B`. AS A REMINDER, the pix2pixHD model requires that the images in domain A are paired with images in domain B; this means that the spatial structure for a pair of images should be similar. For example, domain A could be semantic segmentation maps  and domain B would be the corresponding RGB images, and a pair of images would be the semantic segmentation map of a specific scene and the corresponding RGB image. Because of this requirement, the filenames will need to be the same for image pairs. For example, an image pair would be  `/storage/train_A/0001.png` and `/storage/train_B/0001.png`. Note that you will also need to create a folder, `/storage/checkpoints_dir`, where your model weights and intermediate generated images will be saved during training.
For more information please visit the pix2pix github repository, which includes instructions for training and testing.

Command Format:

`python train.py --name <RUNNAME> --dataroot /storage/example_dataset --checkpoints_dir /storage/checkpoints --label_nc 0 --no_instance`

### Testing pix2pixHD
For testing pix2pixHD, you will need to upload your input data domain A to `/storage/test_A`. You will also need trained network weights, which should be stored in `/storage/checkpoints_dir`.

Command Format:

`python test.py --name <RUNNAME_OF_TRAINED_NETWORK> --dataroot /storage/example_dataset --checkpoints_dir /storage/checkpoints_from_training --results_dir /artifacts/pix2pixhd_testoutputs --resize_or_crop none $@`

<a name="pix2pix"></a>
## Pix2pix for paired image-to-image translation
`pix2pix` is a generative adversarial neural network that transforms one dataset of images, which we refer to as data domain A, into the style of a different dataset, which we refer to as data domain B. Note that the data in domain A must be paired with the data in domain B; this means that the spatial structure of an image in domain A must correpsond to an image in domain D that has the same spatial structure; a good example of this is having domain A be a collection of semantic segmentation maps (each object class is a different color pixel) and domain B is the segmentation maps corresponding RGB image. 


The workspace you can use for both training and testing your model:

https://github.com/dysdsyd/pytorch-CycleGAN-and-pix2pix.git

The pix2pixHD Docker container you can use for processing data:

`okwrtdsh/anaconda3`

### Processing Data
For training create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, etc. In `/path/to/data/A/train`, put training images in style `A`. In `/path/to/data/B/train`, put the corresponding images in style `B`. Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Change the `--fold_A`, `--fold_B` and `--fold_B` to your own domain A's path, domain B's path and output directory path.

Command Format:

`python datasets/combine_A_and_B.py --fold_A /storage/example_dataset/A --fold_B /storage/example_dataset/B --fold_AB /storage/example_dataset/data`

**Note** : We have to run it only once every time there is a change in dataset.

The pix2pixHD Docker container you can use for both training and testing your model:

`daftvader/pix2pix`

### Training pix2pix
Change the `--dataroot`, `--name` and `--checkpoints_dir` to your own dataset's path, model's and checkpoint directory name.

Command Format:

`python train.py --dataroot /storage/example_dataset/data --name experiment --checkpoints_dir /storage/ckp --n_epochs 3 --model pix2pix --netG unet_256 --direction AtoB --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 `


### Testing pix2pix
For testing pix2pix, you will need to upload your input data domain A to `/storage/example_dataset/test_A`. You will also need trained network weights, which should be stored in `/storage/checkpoints_dir`.

Command Format:

`python test.py --dataroot /storage/example_dataset/test_A --name experiment --checkpoints_dir /storage/ckp --results_dir /artifacts --model test --netG unet_256 --direction AtoB  --dataset_mode single --norm batch`

<a name="pggan"></a>
## Progressive growing of GANs (PG-GAN)
PG-GAN functions like a standard GAN framework: the generator neural network takes in a latent noise vector and projects it into the pixel space of RGB images that constitute the 'real' dataset you wish the GAN to model. The Discriminator network determines if its input image is real or fake (i.e, rendered by the Generator network). Each network is influenced by the others error, which trains the Generator to produce highly realistic images. After training, the Discrimiantor network is discarded and the Generator is used to produce novel images that would reasonably come from the training dataset, but do not exist in the training dataset.

In the paperspace persistent storage (using the jupyter notebook) you will need to create the folder `/storage/pggan_dataset` and upload your training image dataset there. This training folder should contain your training images (jpg or png); the naming convention of the images does not matter. For example, a given image named `train_img-1.png` should be located at `/storage/pggan_dataset/train_img-1.png`. They should all be resized and/or cropped to 1024 by 1024 (you can also do 512x512 or attempt 2048x2048; note that the larger images take much longer to run)

The PG-GAN Docker container that you can use for both training and testing the model:

`acarlson32/pytorch-cyclegan-pggan:thirdimage`

The workspace you can use for both training and testing your model: 

`https://github.com/alexacarlson/pytorch_GAN_zoo.git`

### Training PG-GAN
The command format used for training a model:

`python train.py PGAN -c dataset.json -d OUTPUT_DIR -n EXPERIMENT_NAME --no_vis`

where `EXPERIMENT_NAME` is a name you create for your model, and `/storgae/OUTPUT_DIR` is where the training outputs are stored. We recommend that you create the `OUTPUT_DIR` directory in storage so you will be able to access intermediate images produced during the training process, which can take days to weeks depending upon the image size. The `dataset.json` file is part of the PG-GAN training framework that specifies your dataset is located at `/storage/pggan_dataset`, and thus needs to be included in the training command.

### Testing PG-GAN
The command format used for testing an already-trained model; note that `EXPERIMENT_NAME` should match the one you used to train the model. 

`python eval.py visualization -n EXPERIMENT_NAME -m PGAN --dir CHECKPOINT_LOCATION --save_dataset PATH_TO_THE_OUTPUT_DATASET --size_dataset NUMBER_IMAGES_IN_THE_OUTPUT_DATASET --no_vis`

`EXPERIMENT_NAME` is the same name you used in training, `PATH_TO_THE_OUTPUT_DATASET` is where your output images will be saved, and `NUMBER_IMAGES_IN_THE_OUTPUT_DATASET` is the number of images you would like to output, and `CHECKPOINT_LOCATION` is the location of your network checkpoitns/weights in paperspace storage.

<a name="cyclegan"></a>
## CycleGAN for unpaired image-to-image translation:
The paper proposes a method that can capture the characteristics of one image domain and figure out how these characteristics could be translated into another image domain, all in the absence of any paired training examples. CycleGAN uses a special cycle consistency loss to enable training without the need for paired data. In other words, it can translate from one domain to another without a one-to-one mapping between the source and target domain.
This opens up the possibility to do a lot of interesting tasks like photo-enhancement, image colorization, style transfer, etc. All you need is the source and the target dataset (which is simply a directory of images). Check out their github page at <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/README.md> to see some cool examples of what this framework can do!
Your training dataset should have subfolders `trainA`, where you should load your training images for dataset/domain A, `trainB`, where you should load your training images for dataset/domain B, `testA`, where you should load your test images for dataset/domain A  that will be transferred to the style of domain B after training. You will need to create a folder to store the model weights in `/storage`, which is the `CHECKPT_DIR` below.

The cycleGAN Docker container you can use for both training and testing your model:

`acarlson32/pytorch-cyclegan-pggan:thirdimage`

The workspace you can use for both training and testing your model:

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git

### Training cycleGAN
The command format used for training a model:

`python train.py --dataroot DATASET_PATH --name EXPERIMENT_NAME --checkpoints_dir CHECKPT_DIR --load_size RESIZE --crop_size CROP_SIZE --model cycle_gan`

where `DATASET_PATH` is the location of your dataset on papespace, `EXPERIMENT_NAME` is a name you create for your model, 
`CHECKPT_DIR` is the location where the weights and output images will be save, `RESIZE` is a number that your images will be scaled to, `CROP_SIZE` is what the images will be cropped to after scaling. the `--load_size` and `--crop_size` flags are optional.

Command Example:

`python train.py --dataroot /storage/cyclegan-dataset --name train_cyclegan --checkpoints_dir /storage/cgan_checkpts --load_size 256 --crop_size 128 --model cycle_gan --display_id 0`

To see intermediate training image results, check out `CHECKPT_DIR/EXPERIMENT_NAME/web/index.html`

### Testing cycleGAN
The command format used for testing an already-trained model:

`python test.py --dataroot DATASET_PATH --name EXPERIMENT_NAME --checkpoints_dir CHECKPT_DIR --model cycle_gan`

Command Example:

`python test.py --dataroot /storage/cyclegan-dataset --name mapstest_cyclegan --checkpoints_dir /storage/cgan_checkpts --model cycle_gan --results_dir /artifacts`

Note that `EXPERIMENT_NAME` needs to be the same one you used to train the model/generate the weights, similar with `CHECKPT_DIR`.
The test results will be saved to a html file here: `/artifacts/results/EXPERIMENT_NAME/latest_test/index.html`

<a name="stylegan"></a>
## StyleGAN 2
In the paperspace persistent storage (using the jupyter notebook) you will need to create the folder `/storage/stylegan_dataset` and upload your training image dataset there. This training folder should contain your training images (jpg or png); the naming convention of the images does not matter. For example, a given image named `train_img-1.png` should be located at `/storage/style_dataset/train_img-1.png`. They should all be resized and/or cropped to 1024 by 1024 (you can also do 512x512 or attempt 2048x2048; note that the larger images take much longer to run)

The StyleGAN Docker container that you can use for both training and testing the model:

`pytorch/pytorch`

The workspace you can use for both training and testing your model: 

`https://github.com/dysdsyd/stylegan2-pytorch.git`

### Training StyleGAN
The command format used for training a model:

`bash train.sh /storage/DATA EXPERIMENT_NAME /storgae/OUTPUT_DIR IMG_SIZE BATCH_SIZE
`

where `/storage/DATA` is where input data is stored,`EXPERIMENT_NAME` is a name you create for your model, and `/storgae/OUTPUT_DIR` is where the training outputs are stored. We recommend that you create the `OUTPUT_DIR` directory in storage so you will be able to access intermediate images produced during the training process, which can take days to weeks depending upon the image size. `IMG_SIZE` is the size of the image on which you want to train on and `BATCH_SIZE` is the batch size of images for training(decrease it if you run into an OOM error).

### Testing StyleGAN
Updating..

