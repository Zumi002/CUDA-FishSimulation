#include "FishKernels.h"


__global__ void randomizePositionKernel(FishData fd, FishTypes ft, int count, int offset)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= count)
		return;
	//generate random values
	curandState state;
	curand_init(RANDSEED, idx, offset, &state);
	float randomX = curand_uniform(&state);

	float randomY = curand_uniform(&state);

	float randomVX = curand_uniform(&state) * 2 - 1;
	float randomVY = curand_uniform(&state) * 2 - 1;

	//scale values
	randomX = randomX * 2000 - 1000;
	randomY = randomY * 2000 - 1000;

	//apply value
	fd.devPosX[idx + offset] = randomX;
	fd.devPosY[idx + offset] = randomY;
	fd.devVelX[idx + offset] = randomVX;
	fd.devVelY[idx + offset] = randomVY;

	short type = fd.devType[idx + offset];
	fd.devColorRGBA[idx + offset] = ft.color[type];
}

__global__ void setFishTypeKernel(FishData fd, short type, int count, int offset)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= count)
		return;

	fd.devType[idx + offset] = type;
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
	short type = fd.devType[idx];
	float sepRangesq = ft.separateRange[type] * ft.separateRange[type];
	float alignSq = ft.alignRange[type] * ft.alignRange[type];
	float cohSq = ft.coherentRange[type] * ft.coherentRange[type];
	float sepFactor = ft.separateFactor[type];
	float cohFactor = ft.coherentFactor[type];
	float alignFactor = ft.alignFactor[type];
	float obstAvFactor = ft.obstacleAvoidanceFactor[type];

	Speed2D alignSpeed = Speed2D(),
		cohSpeed = Speed2D(),
		sepSpeed = Speed2D(),
		obstAvSpeed = Speed2D();

	float yposavg = 0;
	float xposavg = 0;

	int cohNeigh = 0;
	bool alignNeigh = 0,
		sepNeigh = 0,
		avoidingObst = 0;

	for (int i = 0; i < fishCount; i++)
	{
		if (i == idx)
			continue;

		float otherx = fd.devPosX[i];
		float othery = fd.devPosY[i];
		float dx = posx - otherx;
		float dy = posy - othery;


		float distsq = dx * dx + dy * dy;

		if (distsq <= sepRangesq)
		{
			float dist = sqrt(distsq);
			float d = 1 / (fmaxf(dist, 0.000001f)); //closer fish, bigger force
			sepSpeed.vx += dx*d;
			sepSpeed.vy += dy*d;

			sepNeigh = true;
		}

		if (distsq <= alignSq)
		{
			alignSpeed.vx += fd.devVelX[i];
			alignSpeed.vy += fd.devVelY[i];

			alignNeigh = true;
		}

		if (distsq <= cohSq)
		{
			xposavg += otherx;
			yposavg += othery;

			cohNeigh++;
		}
	}

	if (cohNeigh > 0)
	{
		xposavg /= (float)cohNeigh;
		yposavg /= (float)cohNeigh;

		xposavg -= posx;
		yposavg -= posy;

		cohSpeed = Speed2D(xposavg, yposavg);
	}


	if (mousePos.avoid)
	{
		float mdx = posx - mousePos.x;
		float mdy = posy - mousePos.y;

		float dist = mdy * mdy + mdx * mdx;

		if (dist < 40000)
		{
			obstAvSpeed.vx += mdx;
			obstAvSpeed.vy += mdy;
			avoidingObst = true;
		}
	}

	float minSpeed = ft.minSpeed[type];
	float maxSpeed = ft.maxSpeed[type];

	alignSpeed = steerTowards(alignSpeed, vx, vy, maxSpeed, 0.2f);
	cohSpeed = steerTowards(cohSpeed, vx, vy, maxSpeed, 0.2f);
	sepSpeed = steerTowards(sepSpeed, vx, vy, maxSpeed, 0.2f);
	obstAvSpeed = steerTowards(obstAvSpeed, vx, vy, maxSpeed, 0.2f);

	Speed2D sum = Speed2D(vx, vy);

	//we add steering only when needed 
	if (alignNeigh)
		sum.addScaled(alignSpeed, alignFactor);
	if (cohNeigh)
		sum.addScaled(cohSpeed, cohFactor);
	if (sepNeigh)
		sum.addScaled(sepSpeed, sepFactor);
	if (avoidingObst)
		sum.addScaled(obstAvSpeed, obstAvFactor);

	sum.max(maxSpeed);
	sum.min(minSpeed);

	fd.devColorRGBA[idx] = ft.color[type];

	fd.devTempVelX[idx] = sum.vx;
	fd.devTempVelY[idx] = sum.vy;
}

__global__ void updatePositionKernel(FishData fd, int fishCount)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	float vx = fd.devTempVelX[idx];
	float vy = fd.devTempVelY[idx];

	float posx = fd.devPosX[idx] + vx;
	float posy = fd.devPosY[idx] + vy;


	if (posx > 1000 || posx < -1000)
	{
		vx = -vx;
	}
	if (posy > 1000 || posy < -1000)
	{
		vy = -vy;
	}

	fd.devVelX[idx] = vx;
	fd.devVelY[idx] = vy;

	fd.devPosX[idx] = clamp(posx, -1000, 1000);
	fd.devPosY[idx] = clamp(posy, -1000, 1000);

	//printf("%f %f %f %f %f %f %d\n", fd.devTempVelX[idx], fd.devTempVelY[idx], fd.devVelX[idx], fd.devVelY[idx], fd.devPosX[idx], fd.devPosY[idx], idx);
}

__device__ float clamp(float value, float minVal, float maxVal)
{
	return fminf(fmaxf(value, minVal), maxVal);
}

__device__ Speed2D capSpeed(Speed2D speed2d, float minSpeed, float maxSpeed)
{
	float vx = speed2d.vx;
	float vy = speed2d.vy;

	float speed = sqrt(vx * vx + vy * vy);
	if (speed > 0)
	{
		vx /= speed;
		vy /= speed;
	}

	speed = clamp(speed, minSpeed, maxSpeed);

	return Speed2D(vx * speed, vy * speed);
}

__device__ Speed2D steerTowards(Speed2D speed2d, float vx, float vy, float maxSpeed, float steeringForce)
{
	speed2d.setMag(maxSpeed);
	speed2d.vx -= vx;
	speed2d.vy -= vy;
	speed2d.max(steeringForce);

	return speed2d;
}

__global__ void pauseInteractionsKernel(FishData fd, FishTypes ft, int fishCount)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	fd.devColorRGBA[idx] = ft.color[fd.devType[idx]];
}

__global__ void preGridMakingKernel(FishData fd, FishData tempFd, int fishCount, float gridSize, int cellCount, int collumns)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	float posX = fd.devPosX[idx];
	float posY = fd.devPosY[idx];
	tempFd.devPosX[idx] = posX;
	tempFd.devPosY[idx] = posY;
	tempFd.devVelX[idx] = fd.devVelX[idx];
	tempFd.devVelY[idx] = fd.devVelY[idx];
	tempFd.devType[idx] = fd.devType[idx];
	

	posX += 1000.0f;
	posY += 1000.0f;

	int row = ceilf(fminf(posY,1999) / gridSize)-1;
	int collumn = ceilf(fminf(posX, 1999) / gridSize)-1;
	if (row <= -1)
		row = 0;
	if (collumn <= -1)
		collumn = 0;
	int cell = row * collumns + collumn;

	fd.devGridCell[idx] = cell;
	fd.devID[idx] = idx;
	tempFd.devGridCell[idx] = cell;
	
	//printf("%d %d %d %d %f %f\n",idx, cell, row, collumn, posX, posY);

}

__global__ void postGridMakingKernel(FishData fd, FishData tempFd, int fishCount, int* gridStarts, int* gridEnds)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	int tmpIdx = fd.devID[idx];

	fd.devPosX[idx] = tempFd.devPosX[tmpIdx];
	fd.devPosY[idx] = tempFd.devPosY[tmpIdx];
	fd.devVelX[idx] = tempFd.devVelX[tmpIdx];
	fd.devVelY[idx] = tempFd.devVelY[tmpIdx];
	fd.devType[idx] = tempFd.devType[tmpIdx];
	fd.devGridCell[idx] = tempFd.devGridCell[tmpIdx];

	if (idx == 0)
	{
		gridStarts[fd.devGridCell[idx]] = idx;
	}
	else if (idx == fishCount - 1)
	{
		gridEnds[fd.devGridCell[idx]] = fishCount;
	}
	else if (fd.devGridCell[idx - 1] != fd.devGridCell[idx])
	{
		gridEnds[fd.devGridCell[idx - 1]] = idx;
		gridStarts[fd.devGridCell[idx]] = idx;
	}
}

__global__ void simulateStepGridKernel(FishData fd, FishTypes ft, int fishCount, MousePos mousePos, int* gridStarts, int* gridEnds, int cellCount, int collumns)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= fishCount)
		return;

	float posx = fd.devPosX[idx];
	float posy = fd.devPosY[idx];
	float vx = fd.devVelX[idx];
	float vy = fd.devVelY[idx];
	short type = fd.devType[idx];
	float sepRangesq = ft.separateRange[type] * ft.separateRange[type];
	float alignSq = ft.alignRange[type] * ft.alignRange[type];
	float cohSq = ft.coherentRange[type] * ft.coherentRange[type];
	float sepFactor = ft.separateFactor[type];
	float cohFactor = ft.coherentFactor[type];
	float alignFactor = ft.alignFactor[type];
	float obstAvFactor = ft.obstacleAvoidanceFactor[type];

	Speed2D alignSpeed = Speed2D(),
		cohSpeed = Speed2D(),
		sepSpeed = Speed2D(),
		obstAvSpeed = Speed2D();

	float yposavg = 0;
	float xposavg = 0;

	int cohNeigh = 0;
	bool alignNeigh = 0,
		sepNeigh = 0,
		avoidingObst = 0;

	int myCell = fd.devGridCell[idx];
	int myCol = myCell % collumns;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int cell = myCell + collumns * i + j;

			if (cell < 0 || cell >= cellCount || myCol+j<0 || myCol+j >= collumns)
				continue;

			int k = gridStarts[cell];

			if (k == -1)
				continue;

			while (k<gridEnds[cell])
			{
				if (k == idx)
				{
					k++;
					continue;
				}

				float otherx = fd.devPosX[k];
				float othery = fd.devPosY[k];
				float dx = posx - otherx;
				float dy = posy - othery;


				float distsq = dx * dx + dy * dy;

				if (distsq <= sepRangesq)
				{
					float dist = sqrt(distsq);
					float d = 1 / (fmaxf(dist, 0.000001f)); //closer fish, bigger force
					sepSpeed.vx += dx*d;
					sepSpeed.vy += dy*d;

					sepNeigh = true;
				}

				if (distsq <= alignSq)
				{
					alignSpeed.vx += fd.devVelX[k];
					alignSpeed.vy += fd.devVelY[k];

					alignNeigh = true;
				}

				if (distsq <= cohSq)
				{
					xposavg += otherx;
					yposavg += othery;

					cohNeigh++;
				}
				k++;
			}

		}
	}

	if (cohNeigh > 0)
	{
		xposavg /= (float)cohNeigh;
		yposavg /= (float)cohNeigh;

		xposavg -= posx;
		yposavg -= posy;

		cohSpeed = Speed2D(xposavg, yposavg);
	}


	if (mousePos.avoid)
	{
		float mdx = posx - mousePos.x;
		float mdy = posy - mousePos.y;

		float dist = mdy * mdy + mdx * mdx;

		if (dist < 40000)
		{
			obstAvSpeed.vx += mdx;
			obstAvSpeed.vy += mdy;
			avoidingObst = true;
		}
	}

	float minSpeed = ft.minSpeed[type];
	float maxSpeed = ft.maxSpeed[type];

	alignSpeed = steerTowards(alignSpeed, vx, vy, maxSpeed, 0.2f);
	cohSpeed = steerTowards(cohSpeed, vx, vy, maxSpeed, 0.2f);
	sepSpeed = steerTowards(sepSpeed, vx, vy, maxSpeed, 0.2f);
	obstAvSpeed = steerTowards(obstAvSpeed, vx, vy, maxSpeed, 0.2f);

	Speed2D sum = Speed2D(vx, vy);

	//we add steering only when needed 
	if (alignNeigh)
		sum.addScaled(alignSpeed, alignFactor);
	if (cohNeigh)
		sum.addScaled(cohSpeed, cohFactor);
	if (sepNeigh)
		sum.addScaled(sepSpeed, sepFactor);
	if (avoidingObst)
		sum.addScaled(obstAvSpeed, obstAvFactor);

	sum.max(maxSpeed);
	sum.min(minSpeed);

	fd.devColorRGBA[idx] = ft.color[type];

	fd.devTempVelX[idx] = sum.vx;
	fd.devTempVelY[idx] = sum.vy;
}
