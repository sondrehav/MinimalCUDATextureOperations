#include "program.h"
#include <cstdio>
#include <thread>
#include <string>
#include <map>
#include <cassert>

Program* __program = nullptr;

/* callbacks */

void errorCB(int code, const char* desc)
{
	printf("GLFW error!\n\tcode: %d\n\tdesc: %s\n", code, desc);
	abort();
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	switch (action)
	{
	case GLFW_PRESS:
		return __program->onKeyDown(key, mods);
	case GLFW_RELEASE:
		return __program->onKeyUp(key, mods);
	case GLFW_REPEAT:
		return __program->onKeyPressed(key, mods);
	default:
		return;
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	double x, y;
	glfwGetCursorPos(window, &x, &y);
	switch (action)
	{
	case GLFW_PRESS:
		return __program->onMouseDown(x, y, button, mods);
	case GLFW_RELEASE:
		return __program->onMouseUp(x, y, button, mods);
	}
}
void cursorPosCallback(GLFWwindow* window, double x, double y)
{
	__program->onMouseMove(x, y);
}

void scrollCallback(GLFWwindow* window, double dx, double dy)
{
	__program->onMouseScroll(dx, dy);
}

void windowResizeCallback(GLFWwindow* window, int width, int height)
{
	__program->resized(width, height);
}

void initCallbacks(GLFWwindow* window)
{
	glfwSetKeyCallback(window, keyCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetWindowSizeCallback(window, windowResizeCallback);
}

void destroyCallbacks(GLFWwindow* window)
{
	glfwSetKeyCallback(window, NULL);
	glfwSetMouseButtonCallback(window, NULL);
	glfwSetCursorPosCallback(window, NULL);
	glfwSetScrollCallback(window, NULL);
	glfwSetWindowSizeCallback(window, NULL);
}

/**/

Program::Program(int width, int height) : windowWidth(width), windowHeight(height)
{
	assert(__program == nullptr);
	__program = this;
}

Program::~Program()
{
	assert(__program == this);
	__program = nullptr;
}

void Program::closeWindow()
{
	glfwSetWindowShouldClose(window, true);
}

int Program::run()
{
	

	glfwSetErrorCallback(errorCB);

	/* Initialize the library */
	if (!glfwInit())
	{
		printf("ERROR: Could not initialize GLFW\n");
		return __LINE__;
	}

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(windowWidth, windowHeight, "Hello World", NULL, NULL);
	if (!window)
	{
		printf("ERROR: Could not create GLFW window\n");
		return __LINE__;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	if (gladLoadGL() != 1)
	{
		glfwTerminate();
		printf("ERROR: Could not load OpenGL\n");
		return __LINE__;
	}
	

	init();

	float smoothFPS = maxFps;
	float dt = 0;

	initCallbacks(window);

	while (!glfwWindowShouldClose(window))
	{

		double newMx, newMy;
		glfwGetCursorPos(window, &newMx, &newMy);

		double startTime = glfwGetTime();

		loop(dt);

		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();

		emptyExecQueue();

		float elapsed = glfwGetTime() - startTime;

		if (elapsed < minTime)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds((int)((minTime - elapsed) * 1000)));
		}
		else
		{
			onFrameTimeExceededLimit(elapsed);
		}

		dt = glfwGetTime() - startTime;

		double fps = 1.0 / dt;
		smoothFPS = 0.9 * smoothFPS + 0.1 * fps;

		std::string title = "fps: " + std::to_string((int)round(smoothFPS));
		glfwSetWindowTitle(window, title.c_str());


	}

	emptyExecQueue();

	destroy();

	destroyCallbacks(window);

	glfwTerminate();

	return 0;
}
