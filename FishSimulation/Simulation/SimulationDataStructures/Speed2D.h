#pragma once
#include <cuda_runtime.h>
#include <math.h>

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

	__device__ __host__ Speed2D& operator+=(const Speed2D& b)
	{
		vx += b.vx;
		vy += b.vy;
		return *this;
	}

	__device__ __host__ friend Speed2D operator+(Speed2D a, const Speed2D b)
	{
		return a+=b;
	}

	__device__ __host__ Speed2D& operator-=(const Speed2D& b)
	{
		vx -= b.vx;
		vy -= b.vy;
		return *this;
	}

	__device__ __host__ friend Speed2D operator-(Speed2D a, const Speed2D b)
	{
		return a -= b;
	}

	__device__ __host__ Speed2D& operator*=(const float scale)
	{
		vx *= scale;
		vy *= scale;
		return *this;
	}

	__device__ __host__ friend Speed2D operator*(Speed2D a, const float scale)
	{
		return a *= scale;
	}

	__device__ __host__ float sqMag()
	{
		return vx * vx + vy * vy;
	}

	__device__ __host__ float mag()
	{
		return sqrt(sqMag());
	}
	
	__device__ __host__ Speed2D& setMag(const float scale)
	{
		float len = sqMag();
		if (len > 0)
		{
			len = scale / sqrt(len);
		}
		*this *= len;
		return *this;
	}

	__device__ __host__ Speed2D& max(const float scale)
	{
		float len = sqMag();
		float len2 = scale * scale;
		if (len <= len2)
		{
			return *this;
		}
		setMag(scale);
		return *this;
	}

	__device__ __host__ Speed2D& min(const float scale)
	{
		float len = sqMag();
		float len2 = scale * scale;
		if (len >= len2)
		{
			return *this;
		}
		setMag(scale);
		return *this;
	}

	__device__ __host__ Speed2D& addScaled(const Speed2D a, const float scale)
	{
		*this += a * scale;
		return *this;
	}

};