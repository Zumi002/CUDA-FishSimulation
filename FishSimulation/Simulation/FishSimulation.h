#pragma once

#include <SDL.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "SimulationDataStructures/FishContainers.h"
#include "CudaKernels/FishKernels.h"
#include "SimulationDataStructures/MousePos.h"

#define MAX_FISH_COUNT 1000000

class FishSimulation
{

private:
	const int maxFishCount = MAX_FISH_COUNT;

	int fishCount;
	int maxFishTypes = 10;
	int fishTypesCount = 0;

	bool initialized = false;

	int* devGridStart = nullptr;
	int  cellCount = 0;
	int	 collumns = 0;

	FishData fishData;
	FishData tempFishData;
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
	void makeGrid();
	void allocTempFishData();
public:
	~FishSimulation();
	void addFishType(FishType fishType);
	int getMaxFishCount();
	int getFishCount();
	void setUpSimulation(FishVBOs* fishVBOs);
	void simulationStep();
	void simulationStepGrid();
	void pauseInteractions();
	void randomizePos(int count, int offset);
	int calcBlocksNeeded(int amount, int threadsCount);
	void addFish(int amount, short type);
	FishTypes* getFishTypes();
	int getFishTypeCount();
	MousePos* getMousePos();
	void cleanUp();
};

