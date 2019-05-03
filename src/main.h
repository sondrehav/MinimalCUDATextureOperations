#pragma once

#include "program/program.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/cudaTexture.h"
#include "gl/textureRenderer.h"

class WaveSolver : public Program
{
public:


	WaveSolver()
		: Program(800, 600)
	{
	}


	void init() override;
	void loop(float dt) override;
	void destroy() override;

	void onKeyUp(int keyNum, int mods) override;


protected:
	void onResized(int width, int height) override;
private:
	bool initCUDA();
	float* initialData();

	CUmodule module;
	CUdevice device;
	CUcontext context;

	CUfunction sharpenFunction;

	CUtexref pressureTexRef;
	CUsurfref pressureSurfRef;

	std::unique_ptr<RWCUDATexture2D> textureOdd;
	std::unique_ptr<RWCUDATexture2D> textureEven;
	std::unique_ptr<TextureRenderer> renderer;

	dim3 dimBlock, dimGrid;

	const int width = 512, height = 512;
	bool even = true;

};
