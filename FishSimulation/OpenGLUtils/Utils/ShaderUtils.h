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

		GLuint GetProgramObject();
		GLuint GetVertexShader();
		GLuint GetFragmentShader();
	private:
		GLuint CompileShader(GLuint type, const std::string& source);
		std::string LoadShaderFromFile(const std::string& fileName);

};