#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <algorithm>
#include <string>

#include "helper_math.h"
#include "noise.h"

#define ITERATIONS 1000

#define PRESSURE 1.2
#define SOUND_VELOCITY 340
#define TIME_STEP 7.81e-6
#define STEP_SIZE 3.83e-4

void ppm(float* data, int width, int height, const std::string& path);
void writeOutput(float* data, int width, int height, const std::string& path);

// static __device__ __forceinline__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
// 2D surfaces
surface<void, 2> velocitySurface;
surface<void, 2> pressureSurface;

// --- 2D float4 texture
texture<float2, cudaTextureType2D, cudaReadModeElementType> velocityTexRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> pressureTexRef;

/*
 * Function for transforming to normalized sampler locations.
 */
__device__ __inline__ float2 sl(float x, float y, int width, int height)
{
	return make_float2((x + 0.5f) / (float)width, (y + 0.5f) / (float)height);
}


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



// Simple copy kernel
__global__ void iteratePressure(int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 sampler = sl(x, y, width, height);

		float values = tex2D(pressureTexRef, sampler.x, sampler.y);

		float div = divergence(x, y, width, height);
		float pressure = TIME_STEP * PRESSURE * SOUND_VELOCITY * SOUND_VELOCITY * div / STEP_SIZE;

			// pressure valid
		values -= pressure;

		surf2Dwrite<float>(values, pressureSurface, x * sizeof(float), y);

	}
}

__global__ void iterateVelocity(int width, int height)
{

	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{

		float2 sampler = sl(x, y, width, height);
		float2 values = tex2D(velocityTexRef, sampler.x, sampler.y);
		float2 grad = gradient(x, y, width, height);
		
		float2 velocity = TIME_STEP * grad / (PRESSURE);

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

// Simple copy kernel
__global__ void firstVelocityIteration(int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{

		float2 sampler = sl(x, y, width, height);
		float2 values = tex2D(velocityTexRef, sampler.x, sampler.y);
		float2 grad = gradient(x, y, width, height);

		float2 velocity = 0.5 * TIME_STEP * grad / (PRESSURE);

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

	const size_t width = 256, height = 128;
	float* h_pressureData = new float[width * height];
	float2* h_velocityData = new float2[width * height];

	size_t pressureArraySize = width * height * sizeof(float);
	size_t velocityArraySize = width * height * sizeof(float2);
	/*
	float4 min = make_float4(std::numeric_limits<float>::max());
	float4 max = make_float4(std::numeric_limits<float>::min());
	
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++)
	{
		float4 value = make_float4(
			perlin2d((float)j / width + 8734, (float)i / height + 834, 4.0, 12),
			0,
			0,
			0
		);
		h_data[i * width + j] = value;
		min = fminf(min, value);
		max = fmaxf(max, value);
	}
	for (int i = 0; i < height; i++)
	for (int j = 0; j < width; j++)
	{
		h_data[i * width + j].x = ((h_data[i * width + j] - min) / (max - min) - 0.5).x;
	}*/
	std::fill_n(h_pressureData, width * height, 0.0f);
	h_pressureData[width*(height / 2) + width / 4 + 1] = 5;
	h_pressureData[width*(height / 2 + 1) + width / 4 + 1] = 5;
	h_pressureData[width*(height / 2 + 1) + width / 4] = 5;
	h_pressureData[width*(height / 2) + width / 4] = 5;

	std::fill_n(h_velocityData, width * height, make_float2(0.0f));
	
	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc velocityChannelDesc = cudaCreateChannelDesc<float2>();
	cudaChannelFormatDesc pressureChannelDesc = cudaCreateChannelDesc<float>();

	cudaArray* cuPressureArray;
	CUDA(cudaMallocArray(&cuPressureArray, &pressureChannelDesc, width, height, cudaArraySurfaceLoadStore));

	cudaArray* cuVelocityArray;
	CUDA(cudaMallocArray(&cuVelocityArray, &velocityChannelDesc, width, height, cudaArraySurfaceLoadStore));

	// Copy to device memory some data located at address h_data
	// in host memory 
	CUDA(cudaMemcpyToArray(cuPressureArray, 0, 0, h_pressureData, pressureArraySize, cudaMemcpyHostToDevice));


	CUDA(cudaMemcpyToArray(cuVelocityArray, 0, 0, h_velocityData, velocityArraySize, cudaMemcpyHostToDevice));

	

	// Bind the arrays to the surface references
	CUDA(cudaBindSurfaceToArray(velocitySurface, cuVelocityArray));
	CUDA(cudaBindSurfaceToArray(pressureSurface, cuPressureArray));

	velocityTexRef.normalized = true;
	velocityTexRef.filterMode = cudaFilterModeLinear;
	velocityTexRef.addressMode[0] = cudaAddressModeClamp; // extend at border
	velocityTexRef.addressMode[1] = cudaAddressModeClamp;
	velocityTexRef.addressMode[2] = cudaAddressModeClamp;

	pressureTexRef.normalized = true;
	pressureTexRef.filterMode = cudaFilterModeLinear;
	pressureTexRef.addressMode[0] = cudaAddressModeClamp; // extend at border
	pressureTexRef.addressMode[1] = cudaAddressModeClamp;
	pressureTexRef.addressMode[2] = cudaAddressModeClamp;

	CUDA(cudaBindTextureToArray(&pressureTexRef, cuPressureArray, &pressureChannelDesc));
	CUDA(cudaBindTextureToArray(&velocityTexRef, cuVelocityArray, &velocityChannelDesc));

	// Invoke kernel
	dim3 dimBlock(32, 32);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	firstVelocityIteration << <dimGrid, dimBlock >> > (width, height);
	CUDA(cudaDeviceSynchronize());

	for (int i = 0; i < ITERATIONS; i++)
	{
		iteratePressure<< <dimGrid, dimBlock >> > (width, height);
		CUDA(cudaDeviceSynchronize());

		iterateVelocity<< <dimGrid, dimBlock >> > (width, height);
		CUDA(cudaDeviceSynchronize());
	}
	

	CUDA(cudaPeekAtLastError());

	CUDA(cudaMemcpyFromArray(h_pressureData, cuPressureArray, 0, 0, pressureArraySize, cudaMemcpyDeviceToHost));
	CUDA(cudaMemcpyFromArray(h_velocityData, cuVelocityArray, 0, 0, velocityArraySize, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDA(cudaFreeArray(cuPressureArray));
	CUDA(cudaFreeArray(cuVelocityArray));

	writeOutput(h_pressureData, width, height, "output.ppm");
	
	delete[] h_pressureData;
	delete[] h_velocityData;

	return 0;
}

void writeOutput(float* data, int width, int height, const std::string& path)
{
	float* output = new float[width * height * 3];

	for (int i = 0; i < width * height; i++)
	{
		float value = clamp(data[i] * 0.5f + 0.5f, 0.0f, 1.0f);
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