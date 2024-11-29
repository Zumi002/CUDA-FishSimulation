#include "FishKernels.h"


__global__ void randomizePositionKernel(FishData fd, int count, int offset)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= count)
		return;
	//generate random values
	curandState state;
	curand_init(RANDSEED, idx, offset, &state);
	float randomX = curand_uniform(&state);

	float randomY = curand_uniform(&state);

	//scale values
	randomX = randomX * 200 - 100;
	randomY = randomY * 200 - 100;

	//apply value
	fd.devPosX[idx + offset] = randomX;
	fd.devPosY[idx + offset] = randomY;
}

__global__ void setFishTypeKernel(FishData fd, short type, int count, int offset)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= count)
		return;

	fd.type[idx + offset] = type;
}

__global__ void simulateStepKernel(FishData fd, FishTypes ft, int fishCount, MousePos mousePos)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	float posx = fd.devPosX[idx];
	float posy = fd.devPosY[idx];
	float vx = fd.devVelX[idx];
	float vy = fd.devVelY[idx];

	short type = fd.type[idx];

	float closedx = 0;
	float closedy = 0;
	float sepRangesq = ft.separateRange[type] * ft.separateRange[type];
	float visionSq = ft.alignRange[type] * ft.alignRange[type];
	float sepFactor = ft.separationFactor[type];
	float cohFactor = ft.coherentFactor[type];
	float aligFactor = ft.alignFactor[type];

	float xvelavg = 0;
	float yvelavg = 0;
	float yposavg = 0;
	float xposavg = 0;

	int neigh = 0;
	//get types to shared memory TO-DO

	for (int i = 0; i < fishCount; i++)
	{
		if (i == idx)
			continue;

		float otherx = fd.devPosX[i];
		float othery = fd.devPosY[i];
		float dx = posx - otherx;
		float dy = posy - othery;


		float distsq = dx * dx + dy * dy;

		if (distsq < sepRangesq)
		{
			closedx += dx;
			closedy += dy;
		}

		if (distsq < visionSq)
		{
			xvelavg += fd.devVelX[i];
			yvelavg += fd.devVelY[i];
			xposavg += otherx;
			yposavg += othery;

			neigh++;
		}

	}

	if (neigh > 0)
	{
		xposavg /= (float)neigh;
		yposavg /= (float)neigh;
		xvelavg /= (float)neigh;
		yvelavg /= (float)neigh;

		vx += (xposavg - posx) * cohFactor + (xvelavg - vx) * aligFactor;
		vy += (yposavg - posy) * cohFactor + (yvelavg - vy) * aligFactor;
	}


	vx += closedx * sepFactor;
	vy += closedy * sepFactor;

	//avoid edges
	if (posx < -95.f)
		vx += ft.obstacleAvoidanceFactor[type];
	if (posx > 95.f)
		vx -= ft.obstacleAvoidanceFactor[type];
	if (posy < -95.f)
		vy += ft.obstacleAvoidanceFactor[type];
	if (posy > 95.f)
		vy -= ft.obstacleAvoidanceFactor[type];

	if (mousePos.avoid)
	{
		float mdx = posx - mousePos.x;
		float mdy = posy - mousePos.y;

		float dist = mdy * mdy + mdx * mdx;

		if (dist < 25)
		{
			vx += mdx * ft.obstacleAvoidanceFactor[type];
			vy += mdy * ft.obstacleAvoidanceFactor[type];
		}
	}

	//if (idx==70)
	//	printf("%f %f %f %f %d\n", posx, posy, vx, vy, idx);

	float minspeed = ft.minSpeed[type];
	float maxspeed = ft.maxSpeed[type];

	float speed = sqrt(vx * vx + vy * vy);

	vx /= speed;
	vy /= speed;

	speed = clamp(speed, minspeed, maxspeed);

	fd.devTempVelX[idx] = vx * speed;
	fd.devTempVelY[idx] = vy * speed;

	//printf("%d\n", idx);
}

__global__ void updatePositionKernel(FishData fd, int fishCount)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	float vx = fd.devTempVelX[idx];
	float vy = fd.devTempVelY[idx];

	fd.devVelX[idx] = vx;
	fd.devVelY[idx] = vy;

	fd.devPosX[idx] = clamp(vx + fd.devPosX[idx], -100, 100);
	fd.devPosY[idx] = clamp(vy + fd.devPosY[idx], -100, 100);



	//printf("%f %f %f %f %f %f %d\n", fd.devTempVelX[idx], fd.devTempVelY[idx], fd.devVelX[idx], fd.devVelY[idx], fd.devPosX[idx], fd.devPosY[idx], idx);
}

__device__ float clamp(float value, float minVal, float maxVal) {
	return fminf(fmaxf(value, minVal), maxVal);
}