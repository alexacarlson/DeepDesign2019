#
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from math import *
import pdb 

## path to image folder; need to fill in 
image_folder_path = '/'
list_of_image_files = os.listdir(image_folder_path)

# we can either resize all the images to be the same shape,
new_height = 256 # in pixels
new_width = 256  # in pixels
list_of_resized_images = []
for imgfile in list_of_image_files:
	## load in the image
	full_file_path = os.path.join(image_folder_path, filename)
	temp_img = scipy.misc.imread(full_file_path)
	print "loaded in %s"%(full_file_path)
	#
	if '.pkl' in filename or 'DS_Store' in filename:
		## this skips to the next iteration of the for loop
		continue
	if len(img.shape)>2:
		##skip any corrupted images
		continue
	## resize the image
	new_img = scipy.misc.imresize(img, (new_height, new_width))
	## add in noise to the image
	#h,w,c = new_img.shape
	new_img += 50*np.random.rand(new_img.shape)
	## this saves the augmented image in the same location as the original image
	new_image_filename  = os.path.join(image_folder_path, 'aug_'+list_of_image_fns[0])
    scipy.misc.imsave(new_image_filename, first_image_res_noise)
	del new_img, img
