CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: MasterMultiGPU ResNetCuDNN ResNetCuDNNOpt ResNetMIOpen ResNetMIOpenOpt

MasterMultiGPU: master_multi_gpu.c
	gcc -g master_multi_gpu.c -o MasterMultiGPU

ResNetCuDNN: resnet_cudnn.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn.cu --use_fast_math -lcurand -lcudnn -o ResNetCuDNN

ResNetCuDNNOpt: resnet_cudnn.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn.cu --use_fast_math -lcurand -lcudnn -o  ResNetCuDNNOpt

ResNetMIOpen: resnet_miopen.cpp
	hipcc -g --offload-arch=gfx1100 resnet_miopen.cpp -ffast-math -lMIOpen -lhiprand -o ResNetMIOpen

ResNetMIOpenOpt: resnet_miopen.cpp
	hipcc -O3 --offload-arch=gfx1100 resnet_miopen.cpp -ffast-math -lMIOpen -lhiprand -o ResNetMIOpenOpt

clean:
	rm ResNetCuDNN ResNetCuDNNOpt ResNetMIOpen ResNetMIOpenOpt
