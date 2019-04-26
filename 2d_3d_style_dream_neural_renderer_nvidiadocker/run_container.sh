nvidia-docker run --rm -it \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/:/root/3Dmodels \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/2Dmodels:/root/2Dmodels \
	-v `pwd`:/root \
	tester2
