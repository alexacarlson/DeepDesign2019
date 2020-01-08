from PIL import Image
import numpy as np
import os

colors = [[10,255,10], [0,0,0], [255,0,0], [255, 255, 255], [0, 0, 255]]


def convert_image(path, path2):
	img = Image.open(path).convert('RGB')
	pixels = np.array(list(img.getdata()))
	dists = np.array([np.sum(np.abs(pixels-c), axis=1) for c in colors])
	classes = np.argmin(dists,axis=0).reshape((512,1024)).astype('int64')
	im = Image.fromarray(np.uint8(classes))
	im.save(path2)


def unconvert_image(path, path2):
	img = Image.open(path).convert('L')
	h, w = np.array(img).shape[0:2]
	pixels = np.array(list(img.getdata()))
	pixels_clr = np.array([colors[p] for p in pixels]).reshape((h,w,3))
	im = Image.fromarray(np.uint8(pixels_clr))
	im.save(path2)


#for f in range(0,1201):
for f in range(0,0):
	print(f)
	lab1 = '/Users/gene/Dropbox/sharing/nyc/train_label/%08d.png'%f
	img1 = '/Users/gene/Dropbox/sharing/nyc/train_img/%08d.png'%f
	lab2 = '/Users/gene/Dropbox/sharing/nyc/train_label/%08d.png'%f
	#cmd = 'cp %s %s'%(path1t, path2t)
	#print(cmd)
	#os.system(cmd)
	convert_image(lab1, lab2)

for f in range(1001,1201):
	print(f)
	#lab1 = '/Users/gene/Dropbox/sharing/nyc/train_label/%08d.png'%f
	#lab2 = '/Users/gene/Dropbox/sharing/nyc/train_label_orig/%08d.png'%f
	lab1 = '/Users/gene/Downloads/nyc_1024p/test_latest/images/%06d_input_label.jpg'%f
	lab2 = '/Users/gene/Downloads/nyc_1024p/test_latest/images/%06d_input_label_o.png'%f
	#unconvert_image(lab1, lab2)
	os.system('rm %s'%lab2)