#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include "../SimulationDataStructures/FishContainers.h"
#include "../SimulationDataStructures/MousePos.h"
#include "../SimulationDataStructures/Speed2D.h"

#define RANDSEED 124

//macro from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

__global__ void randomizePositionKernel(FishData fd, FishTypes ft, int count, int offset);

__global__ void simulateStepKernel(FishData fd, FishTypes ft, int fishCount, MousePos pos);

__global__ void setFishTypeKernel(FishData fd, short type, int count, int offset);

__global__ void updatePositionKernel(FishData fd, int fishCount);

__global__ void pauseInteractionsKernel(FishData fd, FishTypes ft, int fishCount);

__global__ void preGridMakingKernel(FishData fd, FishData tempFd, int fishCount, int* gridStarts, float gridSize, int cellCount, int collumns);

__global__ void postGridMakingKernel(FishData fd, FishData tempFd, int fishCount, int* gridStarts);

__global__ void simulateStepGridKernel(FishData fd, FishTypes ft, int fishCount, MousePos pos, int* gridStarts, int cellCount, int collumns);

__device__ float clamp(float value, float minVal, float maxVal);

__device__ Speed2D capSpeed(Speed2D speed2d, float minSpeed, float maxSpeed);

__device__ Speed2D steerTowards(Speed2D speed2d, float vx, float vy, float maxSpeed, float steeringForce);

