#pragma once

#include <SDL.h>
#include <glad/glad.h>

#include "Utils/ShaderUtils.h"
#include "../Simulation/FishSimulation.h"

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_opengl3.h"
#include "../imgui/imgui_impl_sdl2.h"


class GraphicsManager
{

    const std::string defaultVertexShaderFileName 
        = "OpenGLUtils/Shaders/vertexShader.glsl";
    const std::string defaultFragmentShaderFileName
        = "OpenGLUtils/Shaders/fragmentShader.glsl";

    float projection[16] = {
        0.01f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.01f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

	int screenWidth = 640;
	int screenHeight = 480;

	SDL_Window* graphicsWindow = nullptr;
	SDL_GLContext openGLContext = nullptr;
	
	ShaderManager* shManager = nullptr;

    GLuint fishVAO = 0;
    GLuint fishVBO = 0;
    GLuint fishInstancesVBO = 0;

	bool quit = false;

    FishSimulation* simulation;
    FishVBOs* vbos;

    MousePos* mousePos;
	
	void DrawFrame();
	void LoadShaders(const std::string vertexShaderFileName, const std::string fragmentShaderFileName);
    void ConfigureVAO(int maxBoidCount);
    void checkGLError();
    void Input();
    void makeVBO(GLuint& vboRef, int attrID, int maxBoidCount);
    void renderImGUI();
public:
    void Run();
    void CleanUp();
    void CreateGraphicWindow(const std::string windowName);
};

const std::vector<GLfloat> arrowVertices = {
    0.5f,0.0f, // tip
    -0.33f,0.33f,
    -0.33f,-0.33f
};
