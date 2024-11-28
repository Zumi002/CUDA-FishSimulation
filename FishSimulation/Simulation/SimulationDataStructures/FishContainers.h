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
	short* type;
};

struct FishTypes
{
	float* alignRange;
	float* coheherentRange;
	float* separateRange;

	float* alignFactor;
	float* coherentFactor;
	float* separationFactor;
	float* obstacleAvoidanceFactor;
	float* maxSpeed;
	float* minSpeed;
};

struct FishType
{
	float alignRange = 5.f;
	float coheherentRange = 5.f;
	float separateRange = 2.f;

	float alignFactor = 0.1f;
	float coherentFactor = 0.01f;
	float separationFactor = 0.1f;
	float obstacleAvoidanceFactor = 0.2f;
	float maxSpeed = 0.5f;
	float minSpeed = 0.05f;
};

struct FishVBOs
{
	GLuint posXVBO = 0;
	GLuint posYVBO = 0;
	GLuint velXVBO = 0;
	GLuint velYVBO = 0;
	//colorVBO?
};