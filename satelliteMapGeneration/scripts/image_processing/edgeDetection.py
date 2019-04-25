# Apply Canny Edge detection to a set of images
# https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html

import cv2
import argparse
import cv2
import os, os.path
from os.path import basename
import numpy as np
from matplotlib import pyplot as plt

def main():
  print("Applying Canny Edge Detection to all images in: '/%s'" %args.input_dir)
  images = []
  valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

  print("Output folder will be: '/%s'" %args.output_dir)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  for file in os.listdir(args.input_dir):
    extension = os.path.splitext(file)[1]
    if extension.lower() not in valid_image_extensions:
        continue
    images.append(os.path.join(args.input_dir, file))

  print('')
  print('Reading images and starting the edge detection...')
  
  for image in images:
    name = basename(image)
    print('Applying edge detection to', image)
    img = cv2.imread(image, 0)
    if img is None:
        continue

    edges = cv2.Canny(img, 10, 80)    
    cv2.imwrite(os.path.join(args.output_dir, name), edges)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-src', '--input_dir', required=True, dest='input_dir', type=str, help='The folder containing the images from which to get the edges.')
  parser.add_argument('-o', '--output_dir', required=True ,dest='output_dir', type=str, default='.', help='Output directory to save the images')
  args = parser.parse_args()
  main()

