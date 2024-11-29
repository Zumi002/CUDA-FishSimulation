#include "FishSimulation.h"

int FishSimulation::getMaxFishCount()
{
	return maxFishCount;
}

int FishSimulation::getFishCount()
{
	return fishCount;
}

void FishSimulation::setUpSimulation(FishVBOs* fishVBOs)
{
	vbos = fishVBOs;
	fishCount = 0;
	fishData = FishData();
	allocFishTypes();
	addFishType(FishType());
	mapVBOs();
}

void FishSimulation::mapVBOs()
{

	cudaGraphicsResource* crPosX,
		* crPosY,
		* crVelX,
		* crVelY;

	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crPosX, vbos->posXVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crPosY, vbos->posYVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crVelX, vbos->velXVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crVelY, vbos->velYVBO, cudaGraphicsRegisterFlagsNone));

	// Map all resources once
	gpuErrchk(cudaGraphicsMapResources(1, &crPosX, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devPosX, nullptr, crPosX));

	gpuErrchk(cudaGraphicsMapResources(1, &crPosY, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devPosY, nullptr, crPosY));

	gpuErrchk(cudaGraphicsMapResources(1, &crVelX, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devVelX, nullptr, crVelX));

	gpuErrchk(cudaGraphicsMapResources(1, &crVelY, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devVelY, nullptr, crVelY));
	gpuErrchk(cudaMalloc((void**)&fishData.type, maxFishCount * sizeof(short)));
	gpuErrchk(cudaMalloc((void**)&fishData.devTempVelX, maxFishCount * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&fishData.devTempVelY, maxFishCount * sizeof(float)));
	gpuErrchk(cudaMemset(fishData.devTempVelX, 0, maxFishCount * sizeof(float)));
	gpuErrchk(cudaMemset(fishData.devTempVelY, 0, maxFishCount * sizeof(float)));
}

void FishSimulation::allocFishTypes()
{
	fishTypes = new FishTypes();

	fishTypes->alignRange = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->coheherentRange = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->separateRange = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->alignFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->coherentFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->separationFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->obstacleAvoidanceFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->maxSpeed = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->minSpeed = (float*)malloc(maxFishTypes * sizeof(float));

	devfishTypes = FishTypes();
	gpuErrchk(cudaMalloc(&devfishTypes.alignRange, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.coheherentRange, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.separateRange, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.alignFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.coherentFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.separationFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.obstacleAvoidanceFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.maxSpeed, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.minSpeed, maxFishTypes * sizeof(float)));
}

void FishSimulation::syncFishTypes()
{
	gpuErrchk(cudaMemcpy(devfishTypes.alignRange, fishTypes->alignRange, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.coheherentRange, fishTypes->coheherentRange, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.separateRange, fishTypes->separateRange, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.alignFactor, fishTypes->alignFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.coherentFactor, fishTypes->coherentFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.separationFactor, fishTypes->separationFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.obstacleAvoidanceFactor, fishTypes->obstacleAvoidanceFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.maxSpeed, fishTypes->maxSpeed, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.minSpeed, fishTypes->minSpeed, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
}

void FishSimulation::addFishType(FishType fishType)
{
	fishTypes->alignRange[fishTypesCount] = fishType.alignRange;
	fishTypes->coheherentRange[fishTypesCount] = fishType.coheherentRange;
	fishTypes->separateRange[fishTypesCount] = fishType.separateRange;
	fishTypes->alignFactor[fishTypesCount] = fishType.alignFactor;
	fishTypes->coherentFactor[fishTypesCount] = fishType.coherentFactor;
	fishTypes->separationFactor[fishTypesCount] = fishType.separationFactor;
	fishTypes->obstacleAvoidanceFactor[fishTypesCount] = fishType.obstacleAvoidanceFactor;
	fishTypes->maxSpeed[fishTypesCount] = fishType.maxSpeed;
	fishTypes->minSpeed[fishTypesCount] = fishType.minSpeed;

	fishTypesCount++;
}

int FishSimulation::calcBlocksNeeded(int amount, int threadsCount)
{
	return (amount + threadsCount - 1) / threadsCount;
}

void FishSimulation::addFish(int amount, short type)
{
	int blocks = calcBlocksNeeded(amount, 512);

	setFishTypeKernel << <blocks, 512 >> > (fishData, type, amount, fishCount);
	cudaDeviceSynchronize();
	randomizePos(amount, fishCount);
	fishCount += amount;
	
}

int FishSimulation::getFishTypeCount()
{
	return fishTypesCount;
}

FishTypes* FishSimulation::getFishTypes()
{
	return fishTypes;
}

MousePos* FishSimulation::getMousePos()
{
	return mousePos;
}

void FishSimulation::randomizePos(int count, int offset)
{
	int blocks = calcBlocksNeeded(count, 512);

	randomizePositionKernel << <blocks, 512 >> > (fishData, count, offset);
	cudaDeviceSynchronize();
}

void FishSimulation::simulationStep()
{
	int blocks = calcBlocksNeeded(fishCount, 512);
	syncFishTypes();
	simulateStepKernel << <blocks, 512 >> > (fishData, devfishTypes, fishCount, *mousePos);
	cudaDeviceSynchronize();
	updatePositionKernel << <blocks, 512 >> > (fishData, fishCount);
	cudaDeviceSynchronize();
}
