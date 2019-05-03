#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "eventQueue.h"

class Program : public AsyncQueue
{

public:

	Program(int, int);
	virtual ~Program();
	
	virtual void onKeyPressed(int keyNum, int mods) {}
	virtual void onKeyUp(int keyNum, int mods) {}
	virtual void onKeyDown(int keyNum, int mods) {}
	
	virtual void onMouseMove(double x, double y) {}
	virtual void onMouseDown(double x, double y, int mouseNum, int mods) {}
	virtual void onMouseUp(double x, double y, int mouseNum, int mods) {}
	virtual void onMouseScroll(double dx, double dy) {}

	void resized(int width, int height)
	{
		windowWidth = width;
		windowHeight = height;
		onResized(width, height);
	}


	virtual void init() = 0;
	virtual void loop(float dt) = 0;
	virtual void destroy() = 0;

	virtual void onFrameTimeExceededLimit(float) {}

	void closeWindow();

	int run();


	float getMaxFPS() const
	{
		return maxFps;
	}

	void setMaxFPS(float max_fps)
	{
		maxFps = max_fps;
		minTime = 1.0f / maxFps;
	}

protected:
	virtual void onResized(int width, int height) {}
	int windowWidth = 0, windowHeight = 0;

private:

	float maxFps = 60.0f;
	float minTime = 1.0f / maxFps;

	GLFWwindow* window = nullptr;


};
