##
## Construct Architecture Motif dataset
##
#import matplotlib
#matplotlib.use('Agg')

#import keras
#from keras import applications
#import keras_applications
#from keras import backend as K
#import tensorflow as tf
import pdb 
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from scipy.misc import imsave
from PIL import Image
import os
from google_images_download import google_images_download
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='dataset', help="name of the dataset folder to save images default=['pwd'/dataset]")
parser.add_argument("--num_images_per_class", type=int, default=250, help="number of images to save per specified class; note this is not garaunteed and is dependent upon the request/privacy settings for google image search [default: 250]")
parser.add_argument("--classes_file_path", type=str, default='classes.txt', help="specifcy the class you which to optimize within the input image, dependent upon vgg model used [default: dome]")
parser.add_argument("--chromedriver_path", type=str, default='/chromedriver', help="path to location of chromedriver.exe ")


args = parser.parse_args()
num_images_per_class = args.num_images_per_class
dataset_location = args.dataset_name
classes_file_path = args.classes_file_path


## create folder to save dataset
if not os.path.exists(dataset_location):
	os.mkdir(dataset_location)


#class_list = ['stairs', 'stairway',
#				'cinderblocks', 'cinderblock',
#				'stepping_stones', 'stepping_stone',
#				'arches', 'arches_doorways', 
#				'ditches', 'ditch',
#				'fountains', 'fountain',
#				'benches', 'park_bench',
#				'boulders', 'boulder'
#				]

with open(classes_file_path,'r') as ff:
	class_list = [x.strip() for x in ff.readlines()]
	

for classname in class_list:
	if not os.path.exists(os.path.join(dataset_location, classname)):
		os.mkdir(os.path.join(dataset_location, classname))

#
for class_name in class_list:
	print "downloading images for class %s"%(class_name)
	response = google_images_download.googleimagesdownload()
	Arguments = {"keywords":class_name, 
				"limit":num_images_per_class, 
				"type": 'photo',
				"output_directory":dataset_location,
				"image_directory":class_name,
				"no_numbering":True,
				"delay":0.5,
				"chromedriver":args.chromedriver_path}
	absolute_image_paths = response.download(Arguments)
	##
	badfile_counter=0
	for filename in absolute_image_paths[class_name]:
		#print filename
		try:
			## load in image filename
			img = Image.open(filename)
			img.verify()
			del img
		except:
			os.remove(filename)
			badfile_counter+=1
		#pdb.set_trace()
	print 'Found and removed %d corrupted files out of %d total'%(badfile_counter,len(absolute_image_paths))
	#pdb.set_trace()