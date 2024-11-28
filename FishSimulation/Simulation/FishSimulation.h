#pragma once

#include <SDL.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <string>

#include "SimulationDataStructures/FishContainers.h"
#include "CudaKernels/FishKernels.h"

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

	void mapVBOs();
	void allocFishTypes();
	void syncFishTypes();
	void addFishTypes(std::string fileName);
public:
	void addFishType(FishType fishType);
	int getMaxFishCount();
	int getFishCount();
	void setUpSimulation(FishVBOs* fishVBOs);
	void simulationStep();
	void randomizePos(int count, int offset);
	int calcBlocksNeeded(int amount, int threadsCount);
	void addFish(int amount, short type);
	FishTypes* getFishTypes();
	int getFishTypeCount();
};

