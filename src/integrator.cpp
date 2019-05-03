#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <exception>
#include <string>
#include <chrono>
#include <thread>
#include <cuda_gl_interop.h>
#include "integrator.h"
#include "debug.h"
#include "cudaDebug.h"
#include "texture.h"
#include <cuda.h>
#include <cudaGL.h>

#ifdef STANDALONE

Integrator& Integrator::fromGLTexture(const Texture2D& texture, int pmlLayers)
{

	cudaGraphicsResource* resource;
	CUDA(cudaGraphicsGLRegisterImage(&resource, texture.getID(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

	Integrator integrator(texture.width - pmlLayers * 2, texture.height - pmlLayers * 2, pmlLayers);

	

	
}

int run()
{
	if (glfwInit() != GLFW_TRUE) throw std::exception("Could not initialize glfw3.");

	GLFWwindow* window = glfwCreateWindow(800, 600, "CUDA test", NULL, NULL);
	if (window == NULL) throw std::exception("Could not create glfw3window.");

	glfwMakeContextCurrent(window);
	
	if (!gladLoadGL()) throw std::exception("Could not load OpenGL.");

	GL(glViewport(0, 0, 800, 600));

	double maxFPS = 10.0f;
	std::chrono::nanoseconds frameTime((int)(1e9 / maxFPS));

	Texture2D texture(512 + 2 * 64, 512 + 2 * 64, GL_RGBA32F, nullptr);
	
	Integrator integrator = Integrator::fromGLTexture(texture);
	

	while (!glfwWindowShouldClose(window))
	{
		std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

		printf("Refresh!\n");

		GL(glClearColor(0.5, 0.5, 1.0, 1.0));
		GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

		glfwPollEvents();
		glfwSwapBuffers(window);

		std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
		std::chrono::nanoseconds diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

		std::this_thread::sleep_for(frameTime - diff);

	}

	glfwTerminate();
	return 0;
}

int main()
{
	try
	{
		return run();
	} catch (std::exception e)
	{
		printf("Error: %s\n", e.what());
		system("pause");
	}
	return -1;
}

#endif