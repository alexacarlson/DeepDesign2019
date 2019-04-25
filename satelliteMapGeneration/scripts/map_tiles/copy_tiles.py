from random import random
import urllib
import argparse
import os
from os import listdir
from os.path import isfile, join
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument("--in_path", required=True, type=str, help="where images stored")
args = parser.parse_args()


def main():

	in_path = args.in_path
	out_path = '%s/combined'%in_path

	os.system('mkdir %s'%out_path)

	filesS = [f for f in listdir(in_path+"/map") if isfile(join(in_path+"/map", f)) and f != ".DS_Store" and 'map' in f]
	filesM = [f for f in listdir(in_path+"/sat") if isfile(join(in_path+"/sat", f)) and f != ".DS_Store" and 'sat' in f]

	filesC = [f for f in listdir(out_path) if isfile(join(out_path, f)) ]
	nc = len(filesC)

	for f in range(len(filesS)):
		#	idx = int(filesS[f][3:8])
			
		nameS = filesS[f]
		nameM = "sat"+filesS[f][3:]

		if f % 10 == 0:
			print("copied %d / %d"%(f, len(filesS)))
			
		pathS = '%s/%s'%(in_path+"/map/",nameS)
		pathM = '%s/%s'%(in_path+"/sat/",nameM)

		if not isfile(pathM) or not isfile(pathS):
			print("cant find ", nameS, nameM)
			continue

		imgS = Image.open(pathS)
		imgM = Image.open(pathM)

		w, h = imgS.width, imgS.height

		imgC = Image.new('RGB', (2*w, h))
		imgC.paste(imgM, (0, 0))
		imgC.paste(imgS, (w, 0))

		destPath = '%s/%08d.png' % (out_path, nc + f)
		imgC.save(destPath)

		#	imgS3 = imgS.resize((2048, 1024), Image.BICUBIC)
		#	imgM3 = imgM.resize((2048, 1024), Image.BICUBIC)

		#	destS = '%s/%s/%08d.png' % (path1, curr_idx)
		#	destM = '%s/%s/%08d.png' % (out_path, curr_idx)
		#	imgS.save(destS)
		#	imgM.save(destM)


main()