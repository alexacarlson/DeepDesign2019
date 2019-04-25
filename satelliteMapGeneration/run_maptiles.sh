YOUR_API_KEY=pk.eyJ1IjoiYWNhcmxzb24zMiIsImEiOiJjanV2aWx4NzYwMjI3M3lvMGtndXVoYWJ1In0.gFOEVPEMNT63axqe1qtSkA
ROOT_DIR=/home/alexandracarlson/Desktop/Robotics_Building_networks/RoboticsBuilding2019/training-pix2pix

python scripts/map_tiles/get_tiles.py \
  --key $YOUR_API_KEY \
  --width 256 \
  --height 256 \
  --zoom 17 \
  --num_images 400 \
  --output_dir ${ROOT_DIR}/images \
  --augment True \
  --lat_min -33.450000 \
  --lng_min -70.666670 \
  --lat_max -33.577847 \
  --lng_max -70.627689 \
  --style_map cvalenzuela/cjksl3i4f0f212ss77ezlnr6l
