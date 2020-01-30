nvidia-docker run --rm -it \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/:/3Dmodels:ro \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/2Dmodels:/2Dmodels:ro \
	-v `pwd`:/root \
	2d3d-dreamstyle
