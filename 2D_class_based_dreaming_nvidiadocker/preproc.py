import os
import numpy as np
import scipy.misc as sm
import pdb 

def centeredCrop(img, new_height, new_width):
    #
    width =  np.size(img,1)
    height =  np.size(img,0)
    #
    left = int(np.ceil((width - new_width)/2.))
    top = int(np.ceil((height - new_height)/2.))
    right = int(np.floor((width + new_width)/2.))
    bottom = int(np.floor((height + new_height)/2.))
    #print left, top, right, bottom
    cImg = img[top:bottom, left:right]
    return cImg

def centeredPad(img, new_height, new_width):
    #
    width =  np.size(img,1)
    height =  np.size(img,0)
    #
    left = int(np.ceil((new_width - width)/2.))
    top = int(np.ceil((new_height - height)/2.))
    right = int(np.floor((new_width - width)/2.))
    bottom = int(np.floor((new_height - height)/2.))
    #print left, top, right, bottom
    cImg = np.pad(img,((top,bottom),(right,left),(0,0)),'constant')
    #pdb.set_trace()
    return cImg

classes = ['arch', 'bench', 'boulder', 'cinderblock', 'ditch', 'fountain', 'stairs', 'steppingstone']
num_classes = len(classes)

root = 'building_motif_dataset/train'
class_counter = 0 
for class_ in classes:
    class_files = os.listdir(os.path.join(root, class_))
    img_counter = 0
    #
    for fn in class_files:
        try:
            im = sm.imread(os.path.join(root, class_, fn))
        except:
            continue
        #print im.shape
        if len(im.shape)!=3:
            continue
        if im.shape[2]>3:
            im = im[:,:,:3]
            #
        ##img shape
        oh,ow,c = im.shape
        ## adjust height
        des_h = 400
        if im.shape[0]>des_h:
            #print 'crop1'
            im = centeredCrop(im, des_h, ow)
        else:
            im = centeredPad(im,des_h,ow)
            #print 'pad1'
            #
        ## adjust width
        oh1,ow1,c = im.shape
        #print im.shape
        des_w = 600
        if im.shape[1]>des_w:
            im = centeredCrop(im, oh1, des_w)
            #print 'crop2'
        else:
            im = centeredPad(im, oh1, des_w)
            #print 'pad2'
            #
        filename1 = os.path.join(root,'all_images', class_+'_'+str(img_counter)+'.png')
        sm.imsave(filename1, im )
        print filename1
        print im.shape
        #pdb.set_trace()
        img_counter+=1
