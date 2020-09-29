FROM nvidia/cuda:10.2-cudnn7-devel

RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip python3-setuptools make cmake git wget libprotobuf-dev protobuf-compiler libgoogle-glog-dev libopencv-dev \
libhdf5-dev libatlas-base-dev libboost-all-dev libcaffe-cuda-dev

RUN pip3 install --upgrade pip

#for python api
RUN pip3 install numpy opencv-python 

#replace cmake as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

COPY . /openpose

WORKDIR /openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && make -j `nproc`
WORKDIR /openpose

CMD ["/bin/bash"]

