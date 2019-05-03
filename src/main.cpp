#include "main.h"
#include <cudaGL.h>
#include "cudaDebug.h"
#include "debug.h"

void WaveSolver::init()
{
	initCUDA();
}

void WaveSolver::loop(float dt)
{
	const int w = width;
	const int h = height;
	void *args[2] = { (void*)&w, (void*)&h };

	CUDA_D(cuLaunchKernel(sharpenFunction, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, 0, NULL, args, NULL));
	CUDA(cudaDeviceSynchronize());

	if(even)
	{
		renderer->render(textureOdd.get());
		textureOdd->bindToSurfaceRef(module, pressureSurfRef);
		textureEven->bindToTextureRef(module, pressureTexRef);
	} else
	{
		renderer->render(textureEven.get());
		textureOdd->bindToTextureRef(module, pressureTexRef);
		textureEven->bindToSurfaceRef(module, pressureSurfRef);
	}

	even = !even;
}

void WaveSolver::destroy()
{
	textureOdd.reset();
	textureEven.reset();
	renderer.reset();
}

void WaveSolver::onKeyUp(int keyNum, int mods)
{
	if(keyNum == GLFW_KEY_ESCAPE)
	{
		closeWindow();
	}
}


void WaveSolver::onResized(int width, int height)
{
	GL(glViewport(0, 0, windowWidth, windowHeight));
}

bool WaveSolver::initCUDA()
{
	CUDA_D(cuInit(0));

	char name[128];
	CUdevice tempDevice;
	CUDA_D(cuDeviceGet(&tempDevice, 0));
	CUDA_D(cuDeviceGetName(name, 128, tempDevice));
	
	DEBUG_STR("Using CUDA device: ", name);
	
	CUDA_D(cuDeviceGet(&device, tempDevice));
	CUDA_D(cuGLCtxCreate(&context, CU_CTX_SCHED_AUTO, device));
	
	CUDA_D(cuModuleLoad(&module, "ptx/test.ptx"));
	CUDA_D(cuModuleGetFunction(&sharpenFunction, module, "sharpenFunction"));

	GL(glEnable(GL_TEXTURE_2D));
	textureOdd = std::make_unique<RWCUDATexture2D>(width, height, GL_RGBA32F);
	textureEven = std::make_unique<RWCUDATexture2D>(width, height, GL_RGBA32F);

	float* data = initialData();
	textureEven->setData(data, GL_RED, GL_FLOAT);
	delete[] data;

	CUDA_D(cuModuleGetSurfRef(&pressureSurfRef, module, "pressureSurfRef"));
	CUDA_D(cuModuleGetTexRef(&pressureTexRef, module, "pressureTexRef"));

	CUDA_D(cuTexRefSetFilterMode(pressureTexRef, CUfilter_mode::CU_TR_FILTER_MODE_LINEAR));
	CUDA_D(cuTexRefSetAddressMode(pressureTexRef, 0, CUaddress_mode::CU_TR_ADDRESS_MODE_MIRROR));
	CUDA_D(cuTexRefSetAddressMode(pressureTexRef, 1, CUaddress_mode::CU_TR_ADDRESS_MODE_MIRROR));

	textureOdd->bindToSurfaceRef(module, pressureSurfRef);
	textureEven->bindToTextureRef(module, pressureTexRef);

	renderer = std::make_unique<TextureRenderer>();

	dimBlock = dim3(32, 32, 1);
	dimGrid = dim3((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);

	setMaxFPS(60);

	return true;
}

int main()
{
	WaveSolver waveSolverProgram;
	try
	{
		return waveSolverProgram.run();
	} catch(std::exception e)
	{
		DEBUG_STR("Fatal error: \n\n", e.what());
#ifndef _NDEBUG
		__debugbreak();
#endif
		return -1;
	}
}


float* WaveSolver::initialData()
{
	float* data = new float[width * height];
	std::fill_n(data, width * height, 0.0f);
	for(int j = height / 2 - 40; j < height / 2 + 40; j++)
	for(int i = width / 2 - 40; i < width / 2 + 40; i++)
	{
		data[j*width + i] = 1;
	}
	return data;
}
