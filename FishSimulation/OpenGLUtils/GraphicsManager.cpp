#include "GraphicsManager.h"


void GraphicsManager::createGraphicWindow(const std::string windowName)
{
	//Initializing sdl with opengl
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		fprintf(stderr, "SDL cannot initialize video subsytem\n");
		exit(EXIT_FAILURE);
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 2);

	// Creating window
	graphicsWindow = SDL_CreateWindow("Fish simulation", 100, 100,
		screenWidth, screenHeight,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

	if (graphicsWindow == nullptr)
	{
		fprintf(stderr, "SDL window cannot be created\n");
		exit(EXIT_FAILURE);
	}
	openGLContext = SDL_GL_CreateContext(graphicsWindow);
	if (openGLContext == nullptr)
	{
		fprintf(stderr, "Cannot creater openGL context");
		exit(EXIT_FAILURE);
	}

	if (!gladLoadGLLoader(SDL_GL_GetProcAddress))
	{
		fprintf(stderr, "Glad cannot inizialize");
		exit(EXIT_FAILURE);
	}

	// Load shaders
	loadDefaultShaders();

	// Create simulation
	simulation = new FishSimulation();

	// Make vbos
	vbos = new FishVBOs();
	configureVAO(simulation->getMaxFishCount());

	// Setup simulation
	simulation->setUpSimulation(vbos);
	mousePos = simulation->getMousePos();

	// Setup ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForOpenGL(graphicsWindow, openGLContext);
	ImGui_ImplOpenGL3_Init("#version 410");
}

void GraphicsManager::drawFrame()
{
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	SDL_GL_GetDrawableSize(graphicsWindow, &screenWidth, &screenHeight);

	glViewport(0, 0, screenWidth, screenHeight);
	glClearColor(0.086f, 0.086f, 0.113f, 1.f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	GLuint graphicsPipeline = shManager->getProgramObject();

	glUseProgram(graphicsPipeline);

	// Set up projection matrix uniform
	GLint projectionLoc = glGetUniformLocation(graphicsPipeline, "projection");
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, projection);

	// Draw
	glBindVertexArray(fishVAO);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 3, simulation->getFishCount());

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();

	// Create the UI menu
	renderImGUI();

	// Render ImGui
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GraphicsManager::loadShaders(const std::string vertexShaderFileName, const std::string fragmentShaderFileName)
{
	shManager = new ShaderManager(vertexShaderFileName, fragmentShaderFileName);
}

void GraphicsManager::loadDefaultShaders()
{
	shManager = new ShaderManager();
}

void GraphicsManager::cleanUp()
{
	delete vbos;
	delete simulation;

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	SDL_DestroyWindow(graphicsWindow);
	SDL_Quit();
}

void GraphicsManager::checkGLError() {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		std::cerr << "OpenGL error: " << err << std::endl;
	}
}

void GraphicsManager::configureVAO(int maxBoidCount)
{
	// Create and bind VAO first
	glGenVertexArrays(1, &fishVAO);
	glBindVertexArray(fishVAO);

	// Setup vertex buffer (arrow shape)
	glGenBuffers(1, &fishVBO);
	glBindBuffer(GL_ARRAY_BUFFER, fishVBO);
	glBufferData(GL_ARRAY_BUFFER,
		sizeof(GLfloat) * arrowVertices.size(),
		arrowVertices.data(),
		GL_STATIC_DRAW);

	// Setup vertex attributes for arrow vertices
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// Add VBOs
	makeVBO(vbos->posXVBO, 1, maxBoidCount);
	makeVBO(vbos->posYVBO, 2, maxBoidCount);
	makeVBO(vbos->velXVBO, 3, maxBoidCount);
	makeVBO(vbos->velYVBO, 4, maxBoidCount);
	makeVBO(vbos->colorVBO, 5, maxBoidCount);
}

void GraphicsManager::makeVBO(GLuint& vboRef, int attrID, int maxBoidCount)
{
	
	glGenBuffers(1, &vboRef);
	glBindBuffer(GL_ARRAY_BUFFER, vboRef);
	glBufferData(GL_ARRAY_BUFFER, maxBoidCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	//link vbo to vao
	glEnableVertexAttribArray(attrID);
	glVertexAttribPointer(attrID, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glVertexAttribDivisor(attrID, 1);
}

void GraphicsManager::input()
{
	SDL_Event e;

	// Process events
	while (SDL_PollEvent(&e) != 0)
	{
		ImGui_ImplSDL2_ProcessEvent(&e);
		if (e.type == SDL_QUIT)
		{
			quit = true;
		}
	}

	// Handle mouse input
	int x = 0;
	int y = 0;
	Uint32 mouseEvent = SDL_GetMouseState(&x, &y);
	if (mouseEvent & SDL_BUTTON(1))
	{
		mousePos->avoid = true;
		mousePos->x = (float)x / screenWidth * 2000 - 1000;
		mousePos->y = -(float)y / screenHeight * 2000 + 1000;
	}
	else
	{
		mousePos->avoid = false;
	}

}


void GraphicsManager::renderImGUI()
{
	if (simulation == nullptr)
		return;

	static int selectedFishType = 0;    // Currently selected fish type
	static int fishToAdd = 100;         // Number of fish to add
	static float fps = 0.0f;            // FPS display
	static Uint32 oldTime = SDL_GetTicks(); 
	static bool showHelp = false;		
	static bool lang = false;

	// Update FPS dynamically
	Uint32 newTime = SDL_GetTicks();
	fps = 1000.0f / (newTime - oldTime);
	oldTime = newTime;

	FishTypes* fishTypes = simulation->getFishTypes();


	// imGui ui
	ImGui::Begin("Fish Simulation Controls");

	// FPS Counter
	ImGui::Text("FPS: %.1f", fps);
	ImGui::SameLine();
	if (ImGui::Button(pause ? "Resume" : "Pause")) {
		pause = !pause;
	}

	// Aligning help buttton to the right
	float windowWidth = ImGui::GetWindowSize().x;     
	float buttonWidth = ImGui::CalcTextSize("Help").x + ImGui::GetStyle().FramePadding.x * 2.0f; 
	float padding = 20.f;
	ImGui::SameLine(windowWidth - buttonWidth - padding);

	// Button to show help popup
	if (ImGui::Button("Help")) {
		showHelp = true; 
	}

	if (showHelp) {
		// Popup
		ImGui::Begin("Help", &showHelp); 
		if (ImGui::Button("EN"))
		{
			lang = false;
		}
		ImGui::SameLine();
		if (ImGui::Button("PL"))
		{
			lang = true;
		}
		ImGui::Separator();
		ImGui::TextWrapped(lang?helpPopupStringPL.c_str() : helpPopupStringEN.c_str());
		ImGui::End();
	}
	ImGui::Separator();

	// Button to add fish
	ImGui::InputInt("Number of Fish to Add", &fishToAdd);
	if (ImGui::Button("Add Fish")) {
		if (fishToAdd > 0) {
			simulation->addFish(fishToAdd, selectedFishType);
		}
	}

	ImGui::Separator();

	// Dropdown to select fish type
	if (ImGui::BeginCombo("Fish Type", std::to_string(selectedFishType).c_str())) {
		for (int i = 0; i < simulation->getFishTypeCount(); i++) {
			const bool isSelected = (selectedFishType == i);
			if (ImGui::Selectable(std::to_string(i).c_str(), isSelected)) {
				selectedFishType = i;
			}
			if (isSelected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}

	ImGui::Separator();

	// Editable sliders for the selected fish type
	if (selectedFishType >= 0 && selectedFishType < simulation->getFishTypeCount()) {
		ImGui::Text("Adjust Parameters for Fish Type %d", selectedFishType);

		ImGui::SliderFloat("Align Range", &fishTypes->alignRange[selectedFishType], 1.0f, 100.0f);
		ImGui::SliderFloat("Coherent Range", &fishTypes->coherentRange[selectedFishType], 1.0f, 100.0f);
		ImGui::SliderFloat("Separate Range", &fishTypes->separateRange[selectedFishType], 1.0f, 100.0f);

		ImGui::SliderFloat("Align Factor", &fishTypes->alignFactor[selectedFishType], 0.0f, 3.0f);
		ImGui::SliderFloat("Coherent Factor", &fishTypes->coherentFactor[selectedFishType], 0.0f, 3.0f);
		ImGui::SliderFloat("Separation Factor", &fishTypes->separateFactor[selectedFishType], 0.0f, 3.0f);

		ImGui::SliderFloat("Obstacle Avoid Factor", &fishTypes->obstacleAvoidanceFactor[selectedFishType], 0.0f, 3.0f);
		ImGui::SliderFloat("Max Speed", &fishTypes->maxSpeed[selectedFishType], 0.1f, 5.0f);
		ImGui::SliderFloat("Min Speed", &fishTypes->minSpeed[selectedFishType], 0.01f, fishTypes->maxSpeed[selectedFishType]);

		if (fishTypes->maxSpeed[selectedFishType] < fishTypes->minSpeed[selectedFishType])
		{
			fishTypes->minSpeed[selectedFishType] = fishTypes->maxSpeed[selectedFishType];
		}

		static float color[4]; // RGBA values in [0, 1] format
		int& fishColor = fishTypes->color[selectedFishType]; //0xRRGGBBAA format

		// Convert RRGGBBAA to RGBA float
		color[0] = ((fishColor >> 24) & 0xFF) / 255.0f; 
		color[1] = ((fishColor >> 16) & 0xFF) / 255.0f;
		color[2] = ((fishColor >> 8) & 0xFF) / 255.0f; 
		color[3] = (fishColor & 0xFF) / 255.0f;         

		if (ImGui::ColorEdit3("Fish Color", color)) {
			// Convert RGBA float to RRGGBBAA int
			fishColor = ((int)(color[0] * 255) << 24) |
				((int)(color[1] * 255) << 16) |
				((int)(color[2] * 255) << 8) |
				((int)(color[3] * 255));
		}
	}

	ImGui::Separator();

	// Button to add a new fish type
	if (ImGui::Button("Add New Fish Type")) {
		simulation->addFishType(FishType());
		selectedFishType = simulation->getFishTypeCount() - 1; // Automatically select the new type
	}

	ImGui::End();
}

void GraphicsManager::run()
{
	// Check for errors after initializing
	checkGLError();

	// Add first fish
	simulation->addFish(100, 0);

	// Mainloop
	while (!quit)
	{
		// Process input
		input();
		
		// Draw next frame
		drawFrame();

		// Run simulation
		if (!pause)
		{
			simulation->simulationStepGrid();
		}
		else
		{
			simulation->pauseInteractions();
		}

		// Check for cuda and opengl errors
		gpuErrchk(cudaGetLastError());
		checkGLError();

		// Swap buffers
		SDL_GL_SwapWindow(graphicsWindow);
	}

	cleanUp();
}