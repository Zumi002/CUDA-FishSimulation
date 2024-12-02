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



int main(int argc, char* argv[])
{
    GraphicsManager* gm = new GraphicsManager();

    gm->createGraphicWindow("Fish Simulation");
    gm->run();

    delete gm;

    return 0;
}

