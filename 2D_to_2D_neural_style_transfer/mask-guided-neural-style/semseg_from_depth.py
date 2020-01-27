import os
import numpy as np
import scipy.misc
import pdb

images_dir = '/home/alexandracarlson/Desktop/boulder_images/boulder_depth'
image_files = [fn for fn in os.listdir(images_dir) if '._' not in fn]

for fn in image_files:
    img = scipy.misc.imread(os.path.join(images_dir,fn))
    print img.shape
    h,w,c = img.shape
    imgmask = np.zeros(img.shape)
    imgmask[img>5.]=1.
    print os.path.join(images_dir, os.path.splitext(fn)[0]+'_mask'+os.path.splitext(fn)[1])
    scipy.misc.imsave(os.path.join(images_dir, os.path.splitext(fn)[0]+'_mask'+os.path.splitext(fn)[1]), imgmask)

