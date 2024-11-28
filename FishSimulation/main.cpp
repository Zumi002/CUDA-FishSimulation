#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <SDL.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <glad/glad.h>


#include "imgui/imgui.h"
#include "imgui/imgui_impl_sdl2.h"
#include "imgui/imgui_impl_opengl3.h"

#include "OpenGLUtils/Utils/ShaderUtils.h"
#include "OpenGLUtils/GraphicsManager.h"



void SetUpImGUI()
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls    

    // Setup Platform/Renderer backends
    //ImGui_ImplSDL2_InitForOpenGL(gGraphicsWindow, gOpenGLContext);
    ImGui_ImplOpenGL3_Init();
}

void CleanUp()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    //SDL_DestroyWindow(gGraphicsWindow);
    SDL_Quit();
}

int main(int argc, char* argv[])
{
    GraphicsManager* gm = new GraphicsManager();

    gm->CreateGraphicWindow("Fish Simulation");
    gm->Run();

    return 0;
}

