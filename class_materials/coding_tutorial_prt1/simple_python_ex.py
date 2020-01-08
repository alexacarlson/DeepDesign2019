#
# An example of a simple python script 
#
# The hash symbol means that you are making a comment in your code. A comment means
# that, when you run this .py file, python will ignore the statement/not include it as part of the 
# program

# The following are import statements. You use an import statement to load in a module. 
# A module is a file containing Python function definitions.

# os is a module that handles file path manipulations. Another useful module for this is called glob
import os

# Numpy is the core library for scientific computing in Python. 
# It provides a high-performance multidimensional array object, and tools for working with these arrays.
import numpy as np

#  SciPy builds on Numpy arrats, and provides a large number of functions 
# that operate on numpy arrays and are useful for different types of scientific and 
# engineering applications, specifically those with images
import scipy.misc

# cv2 is an image processing module openCV
# import cv2

# matplotlib is a plotting function module 
import matplotlib.pyplot as plt

# math is a basic math functions module (think cos, sin, pi, etc)
from math import *

# PDB, which is a python debugging module. THE MOST USEFUL MODULE EVER.
import pdb 

# Python has a number of basic types including integers, floats, booleans, and strings.
# Integers are whole numbers, e.g., n = 3 assigns the integer 3 to the variable 'n'. 
# You can now use the variable n anywhere in your program in leu of 3.
# Floats are numbers with decimal places, e.g., h = 3.756
# Booleans are logical operators, e.g., y = True
# Strings are usually a bit of text you want to display to someone, e.g., p = 'Hello!'

## functions
#def adds_noise_to_imgs(img):
#	new_img = img +noise
#	return new_img 

## classes
#class car():
#	def self():
#		car_type = 'tesla'
#		num_wheels = 4
#
#	def dynamics(t):
#		fx = x +v*t
#		return fx

# specify the location of your images. This is based on your file system
# note that this data type is a string
#image_folder_path = '/Users/alexandracarlson/Desktop/test_images'
image_folder_path = 'test_images'

# This uses  a function from the module os to get a list of the images in the image_folder_filepath.
# The function returns a list of strings, where each string is the name of one image file.
# Lists are powerful data structures.
list_of_image_fns = os.listdir(image_folder_path)
pdb.set_trace()
# We can check that all the filenames are in the list by printing its length:
print 'A the number of images in our folder %d'%(len(list_of_image_fns))
# and we can print out the first element in the list by indexing into the list. Note that the first element in the list is indexed by 0, which means the last element in a list of 
# length n is n-1. This is also an example of adding strings together to concatenate them.
print 'The first image in the list ' + list_of_image_fns[0]

# this line of code sets a stopping point in the code, which allows you to examine all the
# variables that are in the current work space
pdb.set_trace()

# We now want to read in each image as a numpy array. We are going to each image as an element of a list, and
# use a for loop to iterate over all the elements within list_of_image_fns
list_of_images = []
for filename in list_of_image_fns:
	## we can use an if statement to ignore certain files in the folder
	if '.pkl' in filename or 'DS_Store' in filename:
		# this skips to the next iteration of the for loop
		continue
	# This scipy function takes in the location (filepath) of the image, loads the image into memory, and outputs the image as an 
	# array of pixel intensities
	full_file_path = os.path.join(image_folder_path, filename)
	temp_img = scipy.misc.imread(full_file_path)
	print "loaded in %s"%(full_file_path)
	# add the image to the list
	list_of_images.append(temp_img)
	# delete the contents of the variable 'temp_img' to free up memory. We don't have to do this (python will reassign values in the next iteration 
	# of the for loop) but it is good housekeelping
	del temp_img

pdb.set_trace()
# So now we have images to play around with! 
#Let's start by visualizing the first image using matplotlib.pyplot
# We can access the first image in the list by indexing into the list. 
first_image = list_of_images[0]

# we can check the shape of each image:
print first_image.shape
# see that the image is an array that has three dimensions, height, width, and channels (RGB)
#pdb.set_trace()

#plot the image
print "Plot the first image in the list"
plt.imshow(first_image) # creates a plot object
#plt.show() # displays a plot object
#pdb.set_trace()

# we can index into the image to display a specific color channel, for example, Red. In this example, we 
# are using a tool called array slicing using the colon operator. When we use the colon in the first dimension, 
# we are saying tht we want all the rows of the image in our array slice.
# Similarly, the colon operator used in the second dimension says that we want our slice to include all columns of the image.
plt.imshow(first_image[:,:,0])
plt.title('Plot the red channel of the first image')
#plt.show()
#pdb.set_trace()

# we can also use array slicing to display a patch of the image.
plt.imshow(first_image[:550, 100:550, :])
plt.title('Crop of the first image')
#plt.show()
#pdb.set_trace()

# So now that we can visualize our loaded in data, we can make an array that is (batch_size, img_height, img_width, channels)
# first we need to make sure that each image is the same shape
print "Check out the dimensions of each image"
for img in list_of_images:
	print img.shape
#pdb.set_trace()

# we can either resize all the images to be the same shape,
print("Resizing all images")
new_height = 256 # in pixels
new_width = 256  # in pixels
list_of_resized_images = []
for img in list_of_images:
	## we can use an if statement to ignore certain files in the folder
	if '.pkl' in filename or 'DS_Store' in filename:
		# this skips to the next iteration of the for loop
		continue
	new_img = scipy.misc.imresize(img, (new_height, new_width))
	list_of_resized_images.append(new_img)
	del new_img
#pdb.set_trace()

# or crop the images to that size, assuming each image is larger than (new_height, new_width) in each dimension.
# note that cropping results in image patches.
#for img in list_of_images:
#	# if we want to take the 0:new_size pixels in each dimension
#	new_img = img[:new_height, :new_width]
#	list_of_resized_images.append(new_img)

print "Example of augmenting image by adding random noise"
# We can use cropping to perform data augmentation, which artificially increases our dataset size and adds in variation.
# Other forms of data augmentation include adding random noise into the image:
first_image_res = list_of_resized_images[0]
first_image_res_noise = first_image_res
h,w,c = first_image_res_noise.shape
first_image_res_noise[:,:,0] = first_image_res_noise[:,:,0] + 50*np.random.rand(h,w)
# we can also achieve the same with:
#first_image_res_morered[:,:,0] += 40*np.ones(first_image_res.shape[:1])
# now we need to make sure all the pixels are in the correct range:
first_image_res_noise[first_image_res_noise>255] = 255
first_image_res_noise[first_image_res_noise<0] = 0
#
# see the module scikit-image (import skimage) for more compilicated forms of augmentation, including shifts in exposure, rotating, etc
# we can save our modiied image to our folder:
new_image_filename  = os.path.join(image_folder_path, 'aug_'+list_of_image_fns[0])
scipy.misc.imsave(new_image_filename, first_image_res_noise)
print('Let us check the saved augmented Image!')
#pdb.set_trace()

# now that all of our images are the same size, we can turn the list into a numpy array (batch_size, img_height, img_width, channels)
image_array = np.array(list_of_resized_images)
# verify the dimensions are what we think they should be
print "Print the image array shape"
print image_array.shape
#pdb.set_trace()

# we can use lists to store corresponding labels for each image. Since each image has it's label in the file name, we can parse the filenames
# to generate a list of labels:
print "Print the image labels"
labels = [ filename.split('_')[0] for filename in list_of_image_fns]
labels = []
for filename in list_of_image_fns:
	temp = filename.split('_')[0]
	label.append(temp)
	pdb.set_trace()
print labels
pdb.set_trace()

# to verify that the index of the label is the same as the index of its matching image, lets visualize the third image and its label:
plt.imshow(image_array[2,:,:,:])
plt.title(labels[2])
plt.show()

pdb.set_trace()
## a useful way to store manipulated data is using numpy pickle
import pickle

# write a file
f = open(os.path.join(image_folder_path,"image_example.pkl"), "wb")
pickle.dump(image_array, f)
pickle.dump(labels, f)
f.close()
# read from same file
f = open(os.path.join(image_folder_path,"image_example.pkl"), "rb")
image_array = pickle.load(f)
labels = pickle.load(f)
f.close()

# make sure you've read into the correct variables!
print labels



