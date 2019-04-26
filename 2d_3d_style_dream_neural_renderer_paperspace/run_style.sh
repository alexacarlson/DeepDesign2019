WHICH_TASK=style_transfer_3d
#WHICH_TASK=deep_dream_3d
#WHICH_TASK=neural_renderer

nvidia-docker run --rm -it \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/3Dmodels/:/root/3Dmodels \
	-v /home/alexandracarlson/Desktop/Robotics_Building_networks/2Dmodels:/root/2Dmodels \
	-v `pwd`:/root \
	chainer-cupy-neuralrenderer \
	python neural_renderer/setup.py install &&\
	python neural_renderer/style_transfer_3d/setup.py install &&\
	python neural_renderer/deep_dream_3d/setup.py install && \
	
    python examples/run.py \
    --filename_mesh /root/3Dmodels/TreeCartoon/TreeCartoon1_OBJ/TreeCartoon1_OBJ.obj \
    --filename_style /root/2Dmodels/tree1_res.jpg \
    --filename_output /root/2Dtree_3Dtree
