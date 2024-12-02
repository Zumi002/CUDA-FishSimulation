#pragma once

#include <SDL.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <string>
#include <math.h>
#include <algorithm>

#include "SimulationDataStructures/FishContainers.h"
#include "CudaKernels/FishKernels.h"
#include "SimulationDataStructures/MousePos.h"

#define MAX_FISH_COUNT 100000

class FishSimulation
{

private:
	const int maxFishCount = MAX_FISH_COUNT;

	int fishCount;
	int maxFishTypes = 10;
	int fishTypesCount = 0;

	bool initialized = false;

	FishData fishData;
	FishTypes* fishTypes;
	FishTypes devfishTypes;
	FishVBOs* vbos;

	cudaGraphicsResource* crPosX,
		* crPosY,
		* crVelX,
		* crVelY,
		* crColor;

	MousePos* mousePos = new MousePos(0,0,false);


	void mapVBOs();
	void allocFishTypes();
	void syncFishTypes();
	void unmapVBOs();
	void freeFishTypes();
public:
	~FishSimulation();
	void addFishType(FishType fishType);
	int getMaxFishCount();
	int getFishCount();
	void setUpSimulation(FishVBOs* fishVBOs);
	void simulationStep();
	void pauseInteractions();
	void randomizePos(int count, int offset);
	int calcBlocksNeeded(int amount, int threadsCount);
	void addFish(int amount, short type);
	FishTypes* getFishTypes();
	int getFishTypeCount();
	MousePos* getMousePos();
	void cleanUp();
};

