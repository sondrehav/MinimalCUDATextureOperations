#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <glad/glad.h>

#include "cudaDebug.h"
#include "helper_math.h"
#include "integrator.h"

#define N 64

void generatePMLData(int width, int height, int pmlLayers, float4* data);

//__constant__ struct Params cParams;

// Surfaces for writing
surface<void, 2> velocitySurface;
surface<void, 2> pressureSurface;

// Textures for reading
texture<float2, cudaTextureType2D, cudaReadModeElementType> velocityTexRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> pressureTexRef;

/*
 * Function for transforming to normalized sampler locations.
 */
__device__ __inline__ float2 sl(float x, float y, int width, int height)
{
	return make_float2((x + 0.5f) / (float)width, (y + 0.5f) / (float)height);
}

/*
 * Calculates the divergence of the velocity field. Note that the grid is staggered 
 * and everything that entails...
 */
float __device__ divergence(float x, float y, int width, int height)
{
	float2 origSampler = sl(x, y, width, height);
	float2 xSampler = sl(x + 1.0, y, width, height);
	float2 ySampler = sl(x, y + 1.0, width, height);
	
	float2 valuesOrig = tex2D(velocityTexRef, origSampler.x, origSampler.y);
	float2 valuesX = tex2D(velocityTexRef, xSampler.x, xSampler.y);
	float2 valuesY = tex2D(velocityTexRef, ySampler.x, ySampler.y);
	
	float vx = valuesX.x - valuesOrig.x;
	float vy = valuesY.y - valuesOrig.y;
	
	return vx + vy;
}

/*
 * Calculates the gradient at the specified location.
 */
float2 __device__ gradient(float x, float y, int width, int height)
{
	float2 origSampler = sl(x, y, width, height);
	float2 xSampler = sl(x - 1.0, y, width, height);
	float2 ySampler = sl(x, y - 1.0, width, height);

	float valuesOrig = tex2D(pressureTexRef, origSampler.x, origSampler.y);
	float valuesX = tex2D(pressureTexRef, xSampler.x, xSampler.y);
	float valuesY = tex2D(pressureTexRef, ySampler.x, ySampler.y);

	float vx = valuesOrig - valuesX;
	float vy = valuesOrig - valuesY;
	
	return make_float2(vx, vy);
}

/*
 * Does one iteration of the pressure calculation.
 */
__global__ void iteratePressure(int width, int height, Params cParams)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 sampler = sl(x, y, width, height);

		float values = tex2D(pressureTexRef, sampler.x, sampler.y);

		float div = divergence(x, y, width, height);
		float currentPressure = cParams.timeStep * (cParams.pressure * cParams.soundVelocity * cParams.soundVelocity * div / cParams.stepSize);

		values -= currentPressure;

		surf2Dwrite<float>(values, pressureSurface, x * sizeof(float), y);

	}
}

/*
 * Does one iteration of the velocity calculations. Note that the velocity
 * grid is only defined for width * (height - 1) for the x direction and
 * (width - 1) * height for the y direction due to the staggered grid layout.
 */
__global__ void iterateVelocity(int width, int height, Params cParams)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 sampler = sl(x, y, width, height);
		float2 values = tex2D(velocityTexRef, sampler.x, sampler.y);
		float2 grad = gradient(x, y, width, height);
		
		float2 velocity = cParams.timeStep * (grad / cParams.pressure);

		if (y > 0)
		{
			// x valid
			values.x -= velocity.x;
		}

		if (x > 0)
		{
			// y valid
			values.y -= velocity.y;
		}
		surf2Dwrite<float2>(values, velocitySurface, x * sizeof(float2), y);

	}
}


// Host code
void Integrator::initialize()
{

	// Buffer sizes
	pressureArraySize = params.actualWidth * params.actualHeight * sizeof(float);
	velocityArraySize = params.actualWidth * params.actualHeight * sizeof(float2);

	cuPressureTexture = new CUDATexture2D(params.actualWidth, params.actualHeight, GL_R32F);
	cuVelocityTexture = new CUDATexture2D(params.actualWidth, params.actualHeight, GL_RG32F);

	// Bind the arrays to the surface references

	cuModuleGetSurfRef()
	CUDA_D(cuModuleGetSurfRef(&m_surfWriteRef, m_module, "pressureSurface")
		, "Failed to get surface reference");


	velocityTexRef.normalized = true;
	velocityTexRef.filterMode = cudaFilterModeLinear;
	velocityTexRef.addressMode[0] = cudaAddressModeClamp; // extend at border
	velocityTexRef.addressMode[1] = cudaAddressModeClamp;

	pressureTexRef.normalized = true;
	pressureTexRef.filterMode = cudaFilterModeLinear;
	pressureTexRef.addressMode[0] = cudaAddressModeClamp; // extend at border

	pmlTexRef.normalized = true;
	pmlTexRef.filterMode = cudaFilterModeLinear;
	pmlTexRef.addressMode[0] = cudaAddressModeBorder; // 0 at border
	pmlTexRef.addressMode[1] = cudaAddressModeBorder;
	pmlTexRef.addressMode[2] = cudaAddressModeBorder;

	geometryTexRef.normalized = true;
	geometryTexRef.filterMode = cudaFilterModeLinear;
	geometryTexRef.addressMode[0] = cudaAddressModeBorder; // 0 at border
	geometryTexRef.addressMode[1] = cudaAddressModeBorder;
	geometryTexRef.addressMode[2] = cudaAddressModeBorder;

	CUDA(cudaBindTextureToArray(&pressureTexRef, cuPressureArray, &pressureChannelDesc));
	CUDA(cudaBindTextureToArray(&velocityTexRef, cuVelocityArray, &velocityChannelDesc));
	CUDA(cudaBindTextureToArray(&pmlTexRef, cuPMLArray, &pmlChannelDesc));
	CUDA(cudaBindTextureToArray(&geometryTexRef, cuGeometryArray, &geometryChannelDesc));

	// Invoke kernel
	dimBlock = dim3(32, 32);
	dimGrid = dim3((params.actualWidth + dimBlock.x - 1) / dimBlock.x, (params.actualHeight + dimBlock.y - 1) / dimBlock.y);

	Params tempParams(params);
	tempParams.timeStep = tempParams.timeStep / 2;

	iterateVelocity << <dimGrid, dimBlock >> > (params.actualWidth, params.actualHeight, tempParams);

	CUDA(cudaDeviceSynchronize());
	CUDA(cudaPeekAtLastError());

}

void Integrator::step()
{
	iteratePressure << <dimGrid, dimBlock >> > (params.actualWidth, params.actualHeight, params);
	iterateVelocity << <dimGrid, dimBlock >> > (params.actualWidth, params.actualHeight, params);
	CUDA(cudaPeekAtLastError());
}

void Integrator::process()
{
	while(running)
	{
		std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
		for(int i = 0; i < N; i++)
		{
			step();
		}
		std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
		std::chrono::nanoseconds diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

		double avgDuration = diff.count() / N;
		simulationSpeedNS = simulationSpeedNS * 0.9 + avgDuration * 0.1;
		numIterations += N;
	}
}

void Integrator::destroy()
{

	// Free device memory
	CUDA(cudaFreeArray(cuPressureArray));
	CUDA(cudaFreeArray(cuVelocityArray));
	CUDA(cudaFreeArray(cuPMLArray));

	delete[] h_pressureData;
	delete[] h_pmlData;
	delete[] h_velocityData;

}

void Integrator::writeBackData()
{
	CUDA(cudaMemcpyFromArray(h_pressureData, cuPressureArray, 0, 0, pressureArraySize, cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpyFromArray(h_velocityData, cuVelocityArray, 0, 0, velocityArraySize, cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpyFromArray(h_geometryData, cuGeometryArray, 0, 0, geometryArraySize, cudaMemcpyDeviceToHost));
}

void generatePMLData(int width, int height, int pmlLayers, float4* data)
{
	const size_t actualWidth = width + 2 * pmlLayers, actualHeight = height + 2 * pmlLayers;
	std::fill_n(data, actualWidth * actualHeight, make_float4(0.0f));
	for(int i = 0; i < actualHeight; i++)
	for(int j = 0; j < actualWidth; j++)
	{
		float pressurePMLXDirection = fmaxf((float)(pmlLayers - j) / pmlLayers, 0.0f) + fmaxf((float)j / pmlLayers - (actualWidth - 2.0f - pmlLayers) / pmlLayers, 0.0f);
		float pressurePMLYDirection = fmaxf((float)(pmlLayers - i) / pmlLayers, 0.0f) + fmaxf((float)i / pmlLayers - (actualHeight - 2.0f - pmlLayers) / pmlLayers, 0.0f);
		
		float velocityPMLXDirection = fmaxf((float)(pmlLayers - j) / pmlLayers, 0.0f) + fmaxf((float)j / pmlLayers - (actualWidth - 1.0f - pmlLayers) / pmlLayers, 0.0f);
		float velocityPMLYDirection = fmaxf((float)(pmlLayers - i) / pmlLayers, 0.0f) + fmaxf((float)i / pmlLayers - (actualHeight - pmlLayers) / pmlLayers, 0.0f);

		data[i*actualWidth + j].x = velocityPMLXDirection;
		data[i*actualWidth + j].y = velocityPMLYDirection;
		data[i*actualWidth + j].z = (pressurePMLXDirection + pressurePMLYDirection);

	}
}
