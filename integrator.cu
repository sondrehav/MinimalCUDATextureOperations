#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <algorithm>
#include <string>

#include "helper_math.h"
#include <chrono>
#include <thread>

void ppm(float* data, int width, int height, const std::string& path);
void writeOutput(float* data, int width, int height, const std::string& path);
void generatePMLData(int width, int height, int pmlLayers, double pmlMax, float4* data);

#define ITERATIONS 1000

#define PRESSURE 1.2
#define SOUND_VELOCITY 340
#define TIME_STEP 7.81e-6
#define STEP_SIZE 3.83e-4


// Surfaces for writing
surface<void, 2> velocitySurface;
surface<void, 2> pressureSurface;

// Textures for reading
texture<float2, cudaTextureType2D, cudaReadModeElementType> velocityTexRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> pressureTexRef;
texture<float4, cudaTextureType2D, cudaReadModeElementType> pmlTexRef;

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
__global__ void iteratePressure(int width, int height, double timeStep)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 sampler = sl(x, y, width, height);

		float values = tex2D(pressureTexRef, sampler.x, sampler.y);
		float pmlValue = tex2D(pmlTexRef, sampler.x, sampler.y).z;

		float div = divergence(x, y, width, height);
		float pressure = timeStep * (PRESSURE * SOUND_VELOCITY * SOUND_VELOCITY * div / STEP_SIZE + pmlValue * values);

		values -= pressure;

		surf2Dwrite<float>(values, pressureSurface, x * sizeof(float), y);

	}
}

/*
 * Does one iteration of the velocity calculations. Note that the velocity
 * grid is only defined for width * (height - 1) for the x direction and
 * (width - 1) * height for the y direction due to the staggered grid layout.
 */
__global__ void iterateVelocity(int width, int height, double timeStep)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 sampler = sl(x, y, width, height);
		float2 values = tex2D(velocityTexRef, sampler.x, sampler.y);
		float2 grad = gradient(x, y, width, height);

		float4 pml = tex2D(pmlTexRef, sampler.x, sampler.y);
		float2 pmlVelocity = make_float2(pml.x, pml.y);
		
		float2 velocity = timeStep * (grad / PRESSURE + pmlVelocity * values);

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



/*
 * Checks errors and aborts if something is wrong.
 */
void CUDA(cudaError_t e) { 
	if (e != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorName(e));
		abort();
	}
}

// Host code
int main()
{

	// Simulation parameters
	const size_t pmlLayers = 64;
	const float pmlMax = 1e-1 * 0.5 / TIME_STEP;
	const size_t width = 256, height = 128;

	// Calculate the actual simulation size (size + pmlLayers)
	const size_t actualWidth = width + 2 * pmlLayers, actualHeight= height + 2 * pmlLayers;

	// Buffer sizes
	size_t pressureArraySize = actualWidth * actualHeight * sizeof(float);
	size_t velocityArraySize = actualWidth * actualHeight * sizeof(float2);
	size_t pmlArraySize = actualWidth * actualHeight * sizeof(float4);

	// Allocate the host buffers
	float* h_pressureData = new float[actualWidth * actualHeight];
	float2* h_velocityData = new float2[actualWidth * actualHeight];
	float4* h_pmlData = new float4[actualWidth * actualHeight];

	// Put the stuff in the stuff
	generatePMLData(width, height, pmlLayers, pmlMax, h_pmlData);

	std::fill_n(h_pressureData, actualWidth * actualHeight, 0.0f);
	h_pressureData[actualWidth*(actualHeight / 2) + actualWidth / 4 + 1] = 5;
	h_pressureData[actualWidth*(actualHeight / 2 + 1) + actualWidth / 4 + 1] = 5;
	h_pressureData[actualWidth*(actualHeight / 2 + 1) + actualWidth / 4] = 5;
	h_pressureData[actualWidth*(actualHeight / 2) + actualWidth / 4] = 5;

	std::fill_n(h_velocityData, actualWidth * actualHeight, make_float2(0.0f));
	
	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc velocityChannelDesc = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc pressureChannelDesc = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc pmlChannelDesc = cudaCreateChannelDesc<float4>();

	// Allocate arrays on the device
	cudaArray* cuPressureArray;
	CUDA(cudaMallocArray(&cuPressureArray, &pressureChannelDesc, actualWidth, actualHeight, cudaArraySurfaceLoadStore));

	cudaArray* cuVelocityArray;
	CUDA(cudaMallocArray(&cuVelocityArray, &velocityChannelDesc, actualWidth, actualHeight, cudaArraySurfaceLoadStore));

	cudaArray* cuPMLArray;
	CUDA(cudaMallocArray(&cuPMLArray, &pmlChannelDesc, actualWidth, actualHeight, cudaArraySurfaceLoadStore));

	// Copy to device memory some data located at address h_pressureData in host memory 
	CUDA(cudaMemcpyToArray(cuPressureArray, 0, 0, h_pressureData, pressureArraySize, cudaMemcpyHostToDevice));

	// Copy to device memory some data located at address h_velocityData in host memory 
	CUDA(cudaMemcpyToArray(cuVelocityArray, 0, 0, h_velocityData, velocityArraySize, cudaMemcpyHostToDevice));
	
	// Copy to device memory some data located at address h_pmlData in host memory 
	CUDA(cudaMemcpyToArray(cuPMLArray, 0, 0, h_pmlData, pmlArraySize, cudaMemcpyHostToDevice));

	// Bind the arrays to the surface references
	CUDA(cudaBindSurfaceToArray(velocitySurface, cuVelocityArray));
	CUDA(cudaBindSurfaceToArray(pressureSurface, cuPressureArray));

	velocityTexRef.normalized = true;
	velocityTexRef.filterMode = cudaFilterModeLinear;
	velocityTexRef.addressMode[0] = cudaAddressModeClamp; // extend at border
	velocityTexRef.addressMode[1] = cudaAddressModeClamp;

	pressureTexRef.normalized = true;
	pressureTexRef.filterMode = cudaFilterModeLinear;
	pressureTexRef.addressMode[0] = cudaAddressModeClamp; // extend at border

	pmlTexRef.normalized = true;
	pmlTexRef.filterMode = cudaFilterModeLinear;
	pmlTexRef.addressMode[0] = cudaAddressModeBorder; // extend at border
	pmlTexRef.addressMode[1] = cudaAddressModeBorder; // extend at border
	pmlTexRef.addressMode[2] = cudaAddressModeBorder; // extend at border

	CUDA(cudaBindTextureToArray(&pressureTexRef, cuPressureArray, &pressureChannelDesc));
	CUDA(cudaBindTextureToArray(&velocityTexRef, cuVelocityArray, &velocityChannelDesc));
	CUDA(cudaBindTextureToArray(&pmlTexRef, cuPMLArray, &pmlChannelDesc));

	// Invoke kernel
	dim3 dimBlock(32, 32);
	dim3 dimGrid((actualWidth + dimBlock.x - 1) / dimBlock.x, (actualHeight + dimBlock.y - 1) / dimBlock.y);

	/* FINALLY after all that crap the actual simulation can begin. */
	
	std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

	iterateVelocity << <dimGrid, dimBlock >> > (actualWidth, actualHeight, TIME_STEP * 0.5);
	for (int i = 0; i < ITERATIONS; i++)
	{
		iteratePressure<< <dimGrid, dimBlock >> > (actualWidth, actualHeight, TIME_STEP);
		iterateVelocity<< <dimGrid, dimBlock >> > (actualWidth, actualHeight, TIME_STEP);
		if(i % (ITERATIONS / 100) == 0)
		{
			printf("Iteration %d of %d...\n", i, ITERATIONS);
		}
	}
	
	std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
	std::chrono::nanoseconds diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

	printf("Simulation took %f milliseconds.\n", diff.count() / 1e6);

	CUDA(cudaPeekAtLastError());

	CUDA(cudaMemcpyFromArray(h_pressureData, cuPressureArray, 0, 0, pressureArraySize, cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpyFromArray(h_velocityData, cuVelocityArray, 0, 0, velocityArraySize, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDA(cudaFreeArray(cuPressureArray));
	CUDA(cudaFreeArray(cuVelocityArray));
	CUDA(cudaFreeArray(cuPMLArray));

	writeOutput(h_pressureData, actualWidth, actualHeight, "output.pml");
	
	delete[] h_pressureData;
	delete[] h_pmlData;
	delete[] h_velocityData;

	system("pause");
	return 0;
}

void generatePMLData(int width, int height, int pmlLayers, double pmlMax, float4* data)
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

		data[i*actualWidth + j].x = pmlMax * velocityPMLXDirection;
		data[i*actualWidth + j].y = pmlMax * velocityPMLYDirection;
		data[i*actualWidth + j].z = pmlMax * (pressurePMLXDirection + pressurePMLYDirection);

	}
}

void writeOutput(float* data, int width, int height, const std::string& path)
{
	float* output = new float[width * height * 3];

	for (int i = 0; i < width * height; i++)
	{
		float value = clamp(data[i] * 0.5 + 0.5, 0.0f, 1.0f);
		output[3 * i + 0] = value;
		output[3 * i + 1] = value;
		output[3 * i + 2] = value;
	}

	ppm(output, width, height, "output.ppm");

	delete[] output;
}

void ppm(float* data, int width, int height, const std::string& path)
{

	int i, j;
	FILE *fp = fopen(path.c_str(), "wb"); /* b - binary mode */
	(void)fprintf(fp, "P6\n%d %d\n255\n", width, height);
	for (j = 0; j < height; ++j)
	{
		for (i = 0; i < width; ++i)
		{
			static unsigned char color[3];
			color[0] = (unsigned char) (data[3 * (j * width + i) + 0] * 255);
			color[1] = (unsigned char) (data[3 * (j * width + i) + 1] * 255);
			color[2] = (unsigned char) (data[3 * (j * width + i) + 2] * 255);
			(void)fwrite(color, 1, 3, fp);
		}
	}
	(void)fclose(fp);
}