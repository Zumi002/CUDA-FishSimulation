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
        0.001f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.001f, 0.0f, 0.0f,
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
    bool pause = false;

    FishSimulation* simulation;
    FishVBOs* vbos;

    MousePos* mousePos;
	
	void drawFrame();
	void loadShaders(const std::string vertexShaderFileName, const std::string fragmentShaderFileName);
    void loadDefaultShaders();
    void configureVAO(int maxBoidCount);
    void checkGLError();
    void input();
    void makeVBO(GLuint& vboRef, int attrID, int maxBoidCount);
    void renderImGUI();
public:
    void run();
    void cleanUp();
    void createGraphicWindow(const std::string windowName);
};

const std::vector<GLfloat> arrowVertices = {
    0.5f,0.0f, // tip
    -0.33f,0.33f,
    -0.33f,-0.33f
};

const std::string helpPopupStringEN = R"(
Welcome to the Fish Simulation!

Controls:
- Pause simulation with 'Pause' button
- Use the 'Add Fish' button to spawn more fish of selected type
- Select a fish type from the dropdown to modify it.
- Use sliders to adjust parameters of selected fish type
- Use the color picker to set the fish color
- Use the 'Add new fish type' button to add new fish type

- Hold LMB to make fish swim away from the mouse

)";

const std::string helpPopupStringPL = R"(
Witaj w Symulacji ryb!

Sterowanie:
- Zatrzymaj symulacje za pomoca przyciksu 'Pause'
- Uzyj przycisku 'Add fish', aby dodac wiecej ryb wybranego typu
- Wybierz za pomoca menu typ ryb do modyfikacji 
- Uzyj suwakow aby zmieniac parametry wybranego typu ryb
- Uzyj color pickera, aby zmienic kolor ryb
- Uzyj przycisku 'Add new fish type', aby dodac nowy typ ryb

- Przytrzymaj LPM, aby sprawic by ryby odplywaly od myszy

)";