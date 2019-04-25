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

root_location = os.getcwd()
dataset_name = 'arch_motif_dataset'
dataset_location = os.path.join(root_location, dataset_name)

if not os.path.exists(dataset_location):
	os.mkdir(dataset_location)

num_images_per_class = 1000

class_list = ['stairs', 'stairway',
				'cinderblocks', 'cinderblock',
				'stepping_stones', 'stepping_stone',
				'arches', 'arches_doorways', 
				'ditches', 'ditch',
				'fountains', 'fountain',
				'benches', 'park_bench',
				'boulders', 'boulder'
				]

for classname in class_list:
	if not os.path.exists(os.path.join(dataset_location, classname)):
		os.mkdir(os.path.join(dataset_location, classname))

from google_images_download import google_images_download
for class_name in class_list:
	print "downloading images for class %s"%(class_name)
	response = google_images_download.googleimagesdownload()
	Arguments = {"keywords":class_name, 
				"limit":num_images_per_class, 
				"type": 'photo',
				"output_directory":dataset_location,
				"image_directory":class_name,
				"no_numbering":True,
				"chromedriver":'/home/alexandracarlson/Desktop/chromedriver'}
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