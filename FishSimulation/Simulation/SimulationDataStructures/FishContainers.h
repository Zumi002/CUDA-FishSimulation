#pragma once

#include <glad/glad.h>

struct FishData
{
	float* devPosX;
	float* devPosY;
	float* devVelX;
	float* devVelY;
	float* devTempVelX;
	float* devTempVelY;
	short* devType;
	int*   devColorRGBA;
	int* devGridCell;
	int* devID;
};

struct FishTypes
{
	float* alignRange;
	float* coherentRange;
	float* separateRange;

	float* alignFactor;
	float* coherentFactor;
	float* separateFactor;
	float* obstacleAvoidanceFactor;
	float* maxSpeed;
	float* minSpeed;
	int*   color;
};

struct FishType
{
	float alignRange = 25.0f;
	float coherentRange = 25.0f;
	float separateRange = 20.f;

	float alignFactor = 1.1f;
	float coherentFactor = 1.0f;
	float separateFactor = 1.1f;
	float obstacleAvoidanceFactor = 2.0f;
	float maxSpeed = 3.0f;
	float minSpeed = 1.0f;
	int color = 0x00F0FFFF; //RRGGBBAA
};

struct FishVBOs
{
	GLuint posXVBO = 0;
	GLuint posYVBO = 0;
	GLuint velXVBO = 0;
	GLuint velYVBO = 0;
	GLuint colorVBO = 0;
};