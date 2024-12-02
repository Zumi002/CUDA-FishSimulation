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

	

	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crPosX, vbos->posXVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crPosY, vbos->posYVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crVelX, vbos->velXVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crVelY, vbos->velYVBO, cudaGraphicsRegisterFlagsNone));
	gpuErrchk(cudaGraphicsGLRegisterBuffer(&crColor, vbos->colorVBO, cudaGraphicsRegisterFlagsNone));

	// Map all resources once
	gpuErrchk(cudaGraphicsMapResources(1, &crPosX, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devPosX, nullptr, crPosX));

	gpuErrchk(cudaGraphicsMapResources(1, &crPosY, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devPosY, nullptr, crPosY));

	gpuErrchk(cudaGraphicsMapResources(1, &crVelX, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devVelX, nullptr, crVelX));

	gpuErrchk(cudaGraphicsMapResources(1, &crVelY, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devVelY, nullptr, crVelY));

	gpuErrchk(cudaGraphicsMapResources(1, &crColor, 0));
	gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&fishData.devColorRGBA, nullptr, crColor));

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
	fishTypes->coherentRange = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->separateRange = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->alignFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->coherentFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->separateFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->obstacleAvoidanceFactor = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->maxSpeed = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->minSpeed = (float*)malloc(maxFishTypes * sizeof(float));
	fishTypes->color = (int*)malloc(maxFishTypes * sizeof(int));

	devfishTypes = FishTypes();
	gpuErrchk(cudaMalloc(&devfishTypes.alignRange, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.coherentRange, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.separateRange, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.alignFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.coherentFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.separateFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.obstacleAvoidanceFactor, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.maxSpeed, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.minSpeed, maxFishTypes * sizeof(float)));
	gpuErrchk(cudaMalloc(&devfishTypes.color, maxFishTypes * sizeof(float)));
}

void FishSimulation::syncFishTypes()
{
	gpuErrchk(cudaMemcpy(devfishTypes.alignRange, fishTypes->alignRange, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.coherentRange, fishTypes->coherentRange, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.separateRange, fishTypes->separateRange, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.alignFactor, fishTypes->alignFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.coherentFactor, fishTypes->coherentFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.separateFactor, fishTypes->separateFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.obstacleAvoidanceFactor, fishTypes->obstacleAvoidanceFactor, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.maxSpeed, fishTypes->maxSpeed, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.minSpeed, fishTypes->minSpeed, fishTypesCount * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devfishTypes.color, fishTypes->color, fishTypesCount * sizeof(int), cudaMemcpyHostToDevice));
}

void FishSimulation::addFishType(FishType fishType)
{
	if (fishTypesCount < maxFishTypes)
	{
		fishTypes->alignRange[fishTypesCount] = fishType.alignRange;
		fishTypes->coherentRange[fishTypesCount] = fishType.coherentRange;
		fishTypes->separateRange[fishTypesCount] = fishType.separateRange;
		fishTypes->alignFactor[fishTypesCount] = fishType.alignFactor;
		fishTypes->coherentFactor[fishTypesCount] = fishType.coherentFactor;
		fishTypes->separateFactor[fishTypesCount] = fishType.separateFactor;
		fishTypes->obstacleAvoidanceFactor[fishTypesCount] = fishType.obstacleAvoidanceFactor;
		fishTypes->maxSpeed[fishTypesCount] = fishType.maxSpeed;
		fishTypes->minSpeed[fishTypesCount] = fishType.minSpeed;
		fishTypes->color[fishTypesCount] = fishType.color;

		fishTypesCount++;
	}
}

int FishSimulation::calcBlocksNeeded(int amount, int threadsCount)
{
	return (amount + threadsCount - 1) / threadsCount;
}

void FishSimulation::addFish(int amount, short type)
{
	if (fishCount < maxFishCount)
	{
		amount = std::min(amount, maxFishCount - fishCount);
		int blocks = calcBlocksNeeded(amount, 512);

		setFishTypeKernel << <blocks, 512 >> > (fishData, type, amount, fishCount);
		cudaDeviceSynchronize();
		randomizePos(amount, fishCount);
		fishCount += amount;
	}
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

	randomizePositionKernel << <blocks, 512 >> > (fishData, devfishTypes, count, offset);
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

void FishSimulation::pauseInteractions()
{
	int blocks = calcBlocksNeeded(fishCount, 512);
	syncFishTypes();
	pauseInteractionsKernel << <blocks, 512 >> > (fishData, devfishTypes, fishCount);
	cudaDeviceSynchronize();
}

void FishSimulation::cleanUp()
{
	unmapVBOs();
	freeFishTypes();
	delete fishTypes;
}

void FishSimulation::unmapVBOs()
{
	gpuErrchk(cudaGraphicsUnmapResources(1, &crPosX, 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &crPosY, 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &crVelX, 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &crVelY, 0));
	gpuErrchk(cudaGraphicsUnmapResources(1, &crColor, 0));

	// Unregister VBOs
	gpuErrchk(cudaGraphicsUnregisterResource(crPosX));
	gpuErrchk(cudaGraphicsUnregisterResource(crPosY));
	gpuErrchk(cudaGraphicsUnregisterResource(crVelX));
	gpuErrchk(cudaGraphicsUnregisterResource(crVelY));
	gpuErrchk(cudaGraphicsUnregisterResource(crColor));

	//cudaFree remaining buffers
	gpuErrchk(cudaFree(fishData.type));
	gpuErrchk(cudaFree(fishData.devTempVelX));
	gpuErrchk(cudaFree(fishData.devTempVelY));
}

void FishSimulation::freeFishTypes()
{

	//free host fishTypes
	delete[] fishTypes->alignRange;
	delete[] fishTypes->coherentRange;
	delete[] fishTypes->separateRange;
	delete[] fishTypes->alignFactor;
	delete[] fishTypes->coherentFactor;
	delete[] fishTypes->separateFactor;
	delete[] fishTypes->obstacleAvoidanceFactor;
	delete[] fishTypes->minSpeed;
	delete[] fishTypes->maxSpeed;
	delete[] fishTypes->color;

	//free device fishTypes
	gpuErrchk(cudaFree(devfishTypes.alignRange));
	gpuErrchk(cudaFree(devfishTypes.coherentRange));
	gpuErrchk(cudaFree(devfishTypes.separateRange));
	gpuErrchk(cudaFree(devfishTypes.alignFactor));
	gpuErrchk(cudaFree(devfishTypes.coherentFactor));
	gpuErrchk(cudaFree(devfishTypes.separateFactor));
	gpuErrchk(cudaFree(devfishTypes.obstacleAvoidanceFactor));
	gpuErrchk(cudaFree(devfishTypes.minSpeed));
	gpuErrchk(cudaFree(devfishTypes.maxSpeed));
	gpuErrchk(cudaFree(devfishTypes.color));
}

FishSimulation::~FishSimulation()
{
	cleanUp();
}
