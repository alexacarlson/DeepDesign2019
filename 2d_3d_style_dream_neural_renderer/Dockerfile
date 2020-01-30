
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y python-dev python-pip python-wheel python-setuptools cmake vim imagemagick git
RUN python2.7 -m pip install --upgrade pip && python2.7 -m pip install --upgrade setuptools
RUN python2.7 -m pip install --ignore-installed six \
							cython==0.26.1 \
							scikit-image \
							numpy \
							tqdm 


#ENV CFLAGS=-I/path/to/cudnn/include
#ENV LDFLAGS=-L/path/to/cudnn/lib
#ENV LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV CUDA_PATH=/usr/local/cuda
ENV NVCC=/usr/local/cuda/bin/nvcc 
RUN python2.7 -m pip install --ignore-installed six chainer==3.3.0 cupy==2.3.0

ENV PYTHONPATH=${PYTHONPATH}:/root/neural_renderer:/root/style_transfer_3d:/root/deep_dream_3d

## install neural renderer
#RUN git clone https://github.com/hiroharu-kato/neural_renderer.git
#ADD neural_renderer neural_renderer
#RUN cd neural_renderer && python2.7 setup.py install

## install styletransfer_2d_to_3d
#RUN git clone https://github.com/hiroharu-kato/style_transfer_3d.git
#ADD style_transfer_3d neural_renderer/style_transfer_3d
#RUN cd style_transfer_3d && python2.7 setup.py install

## install deep dream 3D
#RUN git clone https://github.com/hiroharu-kato/deep_dream_3d.git
#ADD deep_dream_3d neural_renderer/deep_dream_3d
#RUN cd deep_dream_3d && python2.7 setup.py install
WORKDIR /root
