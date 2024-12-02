#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <SDL.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <glad/glad.h>
#include <fstream>
#include <sstream>

class ShaderManager
{
	private:
		GLuint programObject;
		GLuint vertexShader;
		GLuint fragmentShader;

	public:
		ShaderManager(const std::string& vertexShaderFileName, const std::string& fragmentShaderFileName);
        ShaderManager();
		GLuint getProgramObject();
		GLuint getVertexShader();
		GLuint getFragmentShader();
	private:
		GLuint compileShader(GLuint type, const std::string& source);
		std::string loadShaderFromFile(const std::string& fileName);

};

const std::string defaultFragmentShader = R"(

#version 410 core

in vec4 vColor;

out vec4 color;

void main()
{
    color = vColor;
}

)";

const std::string defaultVertexShader = R"(

#version 410 core
layout(location = 0) in vec2 vertexPosition;
layout(location = 1) in float posX;           // Instance-specific position X
layout(location = 2) in float posY;           // Instance-specific position Y
layout(location = 3) in float velX;           // Instance-specific velocity X
layout(location = 4) in float velY;           // Instance-specific velocity Y
layout(location = 5) in int intColor;

out vec4 vColor;

uniform mat4 projection;

void main() {


   

    float angle = atan(-velY, velX);
    float c = cos(angle);
    float s = sin(angle);
    mat2 rotation = mat2(c, -s, s, c);

    vec2 rotatedPos = rotation * vertexPosition;
    vec2 finalPos = rotatedPos * 15 + vec2(posX, posY); 

    gl_Position = projection * vec4(finalPos, 0.0, 1.0);

    // Decode color 0xRRGGBBAA to RGBA float
    float r = float((intColor >> 24) & 0xFF) / 255.0f; 
    float g = float((intColor >> 16) & 0xFF) / 255.0f; 
    float b = float((intColor >> 8) & 0xFF) / 255.0f;  
    float a = float(intColor & 0xFF) / 255.0f;         

    // Sk³adanie do vec4
    vColor = vec4(r, g, b, a);
}
)";