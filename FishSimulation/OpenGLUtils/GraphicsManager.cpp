#include "GraphicsManager.h"


void GraphicsManager::CreateGraphicWindow(const std::string windowName)
{
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

	LoadShaders(defaultVertexShaderFileName, defaultFragmentShaderFileName);

	

	simulation = new FishSimulation();
	vbos = new FishVBOs();
	ConfigureVAO(simulation->getMaxFishCount());
	simulation->setUpSimulation(vbos);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForOpenGL(graphicsWindow, openGLContext);
	ImGui_ImplOpenGL3_Init("#version 410");
}

void GraphicsManager::DrawFrame()
{
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	int w, h;

	SDL_GL_GetDrawableSize(graphicsWindow, &w, &h);

	glViewport(0, 0, w, h);
	glClearColor(0.086f, 0.086f, 0.113f, 1.f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	GLuint graphicsPipeline = shManager->GetProgramObject();

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

void GraphicsManager::LoadShaders(const std::string vertexShaderFileName, const std::string fragmentShaderFileName)
{
	shManager = new ShaderManager(vertexShaderFileName, fragmentShaderFileName);
}

void GraphicsManager::CleanUp()
{
	SDL_DestroyWindow(graphicsWindow);
	SDL_Quit();
}

void GraphicsManager::checkGLError() {
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR) {
		std::cerr << "OpenGL error: " << err << std::endl;
	}
}

void GraphicsManager::ConfigureVAO(int maxBoidCount)
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

	makeVBO(vbos->posXVBO, 1, maxBoidCount);
	makeVBO(vbos->posYVBO, 2, maxBoidCount);
	makeVBO(vbos->velXVBO, 3, maxBoidCount);
	makeVBO(vbos->velYVBO, 4, maxBoidCount);
}

void GraphicsManager::makeVBO(GLuint& vboRef, int attrID, int maxBoidCount)
{
	glGenBuffers(1, &vboRef);
	glBindBuffer(GL_ARRAY_BUFFER, vboRef);
	glBufferData(GL_ARRAY_BUFFER, maxBoidCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(attrID); // Attribute location 4
	glVertexAttribPointer(attrID, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glVertexAttribDivisor(attrID, 1);
}

void GraphicsManager::Input()
{
	SDL_Event e;
	while (SDL_PollEvent(&e) != 0)
	{
		ImGui_ImplSDL2_ProcessEvent(&e);
		if (e.type == SDL_QUIT)
		{
			quit = true;
		}
	}

}

void GraphicsManager::renderImGUI()
{
	static int selectedFishType = 0;    // Currently selected fish type
	static int fishToAdd = 100;         // Number of fish to add
	static float fps = 0.0f;            // FPS display
	static Uint32 oldTime = SDL_GetTicks();

	// Update FPS dynamically
	Uint32 newTime = SDL_GetTicks();
	fps = 1000.0f / (newTime - oldTime);
	oldTime = newTime;

	FishTypes* fishTypes = simulation->getFishTypes();

	ImGui::Begin("Fish Simulation Controls");

	// FPS Counter
	ImGui::Text("FPS: %.1f", fps);

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

		ImGui::SliderFloat("Align Range", &fishTypes->alignRange[selectedFishType], 1.0f, 10.0f);
		ImGui::SliderFloat("Coherent Range", &fishTypes->coheherentRange[selectedFishType], 1.0f, 10.0f);
		ImGui::SliderFloat("Separate Range", &fishTypes->separateRange[selectedFishType], 1.0f, 10.0f);

		ImGui::SliderFloat("Align Factor", &fishTypes->alignFactor[selectedFishType], 0.0f, 1.0f);
		ImGui::SliderFloat("Coherent Factor", &fishTypes->coherentFactor[selectedFishType], 0.0f, 1.0f);
		ImGui::SliderFloat("Separation Factor", &fishTypes->separationFactor[selectedFishType], 0.0f, 1.0f);

		ImGui::SliderFloat("Obstacle Avoid Factor", &fishTypes->obstacleAvoidanceFactor[selectedFishType], 0.0f, 1.0f);
		ImGui::SliderFloat("Max Speed", &fishTypes->maxSpeed[selectedFishType], 0.1f, 5.0f);
		ImGui::SliderFloat("Min Speed", &fishTypes->minSpeed[selectedFishType], 0.01f, 1.0f);
	}

	ImGui::Separator();

	// Button to add a new fish type
	if (ImGui::Button("Add New Fish Type")) {
		simulation->addFishType(FishType());
		selectedFishType = simulation->getFishTypeCount() - 1; // Automatically select the new type
	}

	ImGui::End();
}

void GraphicsManager::Run()
{
	//mainloop
	checkGLError();
	simulation->addFish(1000,0);

	
	while (!quit)
	{
		
		Input();
		DrawFrame();
		simulation->simulationStep();
		gpuErrchk(cudaGetLastError());
		checkGLError();
		SDL_GL_SwapWindow(graphicsWindow);
	}
}