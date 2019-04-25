from random import random
import urllib
import argparse
import os
import random
from PIL import Image
from cStringIO import StringIO
from io import BytesIO
import requests
import pdb 

def get_style(style, location, zoom, width, height, ext):
	'''
	Get the Mabpox style url
	'''
	lng, lat = location
	#url = 'https://api.mapbox.com/styles/v1/%s/static/%f,%f,%d,0,0/%dx%d?access_token=%s'%(style, lat, lng, zoom, width, height, args.key)
	#url = 'https://api.mapbox.com/styles/v1/%s/static/%f,%f,%d/%dx%d?access_token=%s'%(style, lat, lng, zoom, width, height, args.key)
	## askc fix
	url = 'https://api.mapbox.com/v4/mapbox.satellite/%f,%f,%d/%dx%d.jpg?access_token=%s'%(lat, lng, zoom, width, height, args.key)
	#url = 'https://api.mapbox.com/v4/%s/%f,%f,%d/%dx%d.%s?access_token=%s'%(style, lat, lng, zoom, width, height, ext, args.key)
	#url = 'https://api.tiles.mapbox.com/v4/%s/%f,%f,%d/%dx%d.%s?access_token=%s'%(style, lat, lng, zoom, width, height, ext, args.key)
	
	#url = 'https://api.mapbox.com/styles/v1/%s/static/%f,%f,%d,0,0/%dx%d?access_token=%s'%(style, lat, lng, zoom, width, height, args.key)
	return url
	
def download_map_sat(dir_out, t, lat, lng, zoom, out_w, out_h):
	'''
	Download images
	'''
	#path_map = "%s/map/map%05d_%f,%f.%s"%(dir_out, t,lat,lng, 'jpg')
	path_sat = "%s/sat/sat%05d_%f,%f.%s"%(dir_out, t,lat,lng, 'jpg')
	w, h = out_w, out_h
	if args.augment:
		w, h = min(1280, 1.5*out_w), min(1280, 1.5*out_h)
	
	#url_map = get_style(args.style_map, (lat, lng), zoom, int(w), int(h), 'jpg')
	url_sat = get_style(args.style_sat, (lat, lng), zoom, int(w), int(h), 'jpg')
	#urllib.urlretrieve(url_map, path_map)
	urllib.urlretrieve(url_sat, path_sat)
	#with open(path_map,'wb') as img:
	#	img.write(requests.get(url_map).content)
	#with open(path_sat,'wb') as img:
	#	img.write(requests.get(url_sat).content)
	#pdb.set_trace()
	
	if args.augment:
		#img_map = Image.open(path_map)
		img_sat = Image.open(path_sat)
		
		ang = -19 + 38*random.random()
		
		#img_map = img_map.rotate(ang, resample=Image.BICUBIC, expand=False)
		img_sat = img_sat.rotate(ang, resample=Image.BICUBIC, expand=False)
		
		#x1, y1 = int((img_map.width-out_w)*0.5), int((img_map.height-out_h)*0.5)
		x1, y1 = int((img_sat.width-out_w)*0.5), int((img_sat.height-out_h)*0.5)
		
		#img_map = img_map.crop((x1, y1, x1+out_w, y1+out_h))
		img_sat = img_sat.crop((x1, y1, x1+out_w, y1+out_h))

		flipv = random.random() < 0.25
		fliph = random.random() < 0.25

		if flipv:
			#img_map = img_map.transpose(Image.FLIP_TOP_BOTTOM)
			img_sat = img_sat.transpose(Image.FLIP_TOP_BOTTOM)
		if fliph:
			#img_map = img_map.transpose(Image.FLIP_LEFT_RIGHT)
			img_sat = img_sat.transpose(Image.FLIP_LEFT_RIGHT)

		#img_map.save(path_map)
		img_sat.save(path_sat)

def main():
	w, h = args.width, args.height
	n = args.num_images
	zoom = args.zoom
	output_dir = args.output_dir

	os.system('mkdir %s'%output_dir)
	os.system('mkdir %s/map'%output_dir)
	os.system('mkdir %s/sat'%output_dir)
	
	for t in range(n):
		lng = args.lng_min + (args.lng_max - args.lng_min) * random.random()
		lat = args.lat_min + (args.lat_max - args.lat_min) * random.random()
		print('Getting map image%s, %s, %s. (%s/%s)' % (lat, lng, zoom, t, n))
		download_map_sat(output_dir, t, lat, lng, zoom, w, h)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--key", required=True, help="API key")
	parser.add_argument("--width", default=512, required=True, type=int, help="width")
	parser.add_argument("--height", default=512, required=True, type=int, help="height")
	parser.add_argument("--zoom", default=17, required=True, type=int, help="zoom")
	parser.add_argument("--num_images", required=True, type=int, help="num images")
	parser.add_argument("--output_dir", required=True, type=str, help="where to save images")
	parser.add_argument("--augment", required=True, default=False, type=bool, help="rotate (augment) images?")
	#parser.add_argument("--style_map", required=True, type=str, help="Mapbox style to use. Format should be: user/mapID")
	#parser.add_argument("--style_map", required=False, type=str, default='mapbox/streets-v11', help="Satellite style to use")
	#parser.add_argument("--style_sat", required=False, type=str, default='mapbox/satellite-v9', help="Satellite style to use")
	## askc fix
	parser.add_argument("--style_map", required=False, type=str, default='mapbox.streets', help="Mapbox style to use. Format should be: user/mapID")
	parser.add_argument("--style_sat", required=False, type=str, default='mapbox.satellite', help="Satellite style to use")
	parser.add_argument("--lat_min", required=True, type=float, help="Min latitude")
	parser.add_argument("--lng_min", required=True, type=float, help="Min longitude")
	parser.add_argument("--lat_max", required=True, type=float, help="Max latitude")
	parser.add_argument("--lng_max", required=True, type=float, help="Max longitude")
	args = parser.parse_args()
	main()
