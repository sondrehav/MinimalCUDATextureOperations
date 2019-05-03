#pragma once
#include <cuda_runtime.h>
#include <thread>
#include "texture.h"

class Texture2D;

struct Params
{
	Params(size_t width, size_t height, size_t pmlLayers)
		: width(width),
		  height(height),
		  pmlLayers(pmlLayers)
	{
	}

	const size_t width;
	const size_t height;
	const size_t pmlLayers;

	const size_t actualWidth = width + pmlLayers * 2;
	const size_t actualHeight = height + pmlLayers * 2;

	float pressure = 1.2f;
	float soundVelocity = 340.0f;
	float stepSize = 3.83e-4;
	float timeStep = 7.81e-6;

	float pmlMax = 1e-1 * 0.5 / timeStep;
};

class Integrator
{
	
public:

	static Integrator& fromGLTexture(const Texture2D& texture, int pmlLayers);

	Integrator(size_t width, size_t height, size_t pmlLayers) : params(width, height, pmlLayers)
	{
		initialize();
	}

	Integrator(Params params) : params(params)
	{
		initialize();
	}

	~Integrator()
	{
		destroy();
	}

	const Params& getParams() { return params; }

	void step();

private:
	void initialize();
	void destroy();

	Params params;

	size_t pressureArraySize;
	size_t velocityArraySize;

	CUDATexture2D* cuPressureTexture = nullptr;
	CUDATexture2D* cuVelocityTexture = nullptr;

	dim3 dimBlock;
	dim3 dimGrid;

};
