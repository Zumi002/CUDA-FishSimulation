#pragma once
#include <cuda_runtime.h>

struct Speed2D
{
	float vx;
	float vy;

	__device__ __host__ Speed2D(float x, float y)
	{
		vx = x;
		vy = y;
	}
	__device__ __host__ Speed2D()
	{
		vx = 0;
		vy = 0;
	}
};