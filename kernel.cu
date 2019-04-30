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
void writeOutput(float4* data, int width, int height, const std::string& path);

// static __device__ __forceinline__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;

// --- 2D float4 texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;

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
	
	float4 valuesOrig = tex2D(texRef, origSampler.x, origSampler.y);
	float4 valuesX = tex2D(texRef, xSampler.x, xSampler.y);
	float4 valuesY = tex2D(texRef, ySampler.x, ySampler.y);
	
	float vx = valuesX.y - valuesOrig.y;
	float vy = valuesY.z - valuesOrig.z;
	
	return vx + vy;
}

float2 __device__ gradient(float x, float y, int width, int height)
{
	float2 origSampler = sl(x, y, width, height);
	float2 xSampler = sl(x - 1.0, y, width, height);
	float2 ySampler = sl(x, y - 1.0, width, height);

	float4 valuesOrig = tex2D(texRef, origSampler.x, origSampler.y);
	float4 valuesX = tex2D(texRef, xSampler.x, xSampler.y);
	float4 valuesY = tex2D(texRef, ySampler.x, ySampler.y);

	float vx = valuesOrig.x - valuesX.x;
	float vy = valuesOrig.x - valuesY.x;
	
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

		float4 values = tex2D(texRef, sampler.x, sampler.y);

		float div = divergence(x, y, width, height);
		float pressure = TIME_STEP * PRESSURE * SOUND_VELOCITY * SOUND_VELOCITY * div / STEP_SIZE;

			// pressure valid
		values.x -= pressure;

		surf2Dwrite<float4>(values, outputSurfRef, x * sizeof(float4), y);

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
		float4 values = tex2D(texRef, sampler.x, sampler.y);
		float2 grad = gradient(x, y, width, height);
		
		float2 velocity = TIME_STEP * grad / (PRESSURE);

		if (y > 0)
		{
			// x valid
			values.y -= velocity.x;
		}

		if (x > 0)
		{
			// y valid
			values.z -= velocity.y;
		}
		surf2Dwrite<float4>(values, outputSurfRef, x * sizeof(float4), y);

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
		float4 values = tex2D(texRef, sampler.x, sampler.y);
		float2 grad = gradient(x, y, width, height);

		float2 velocity = 0.5 * TIME_STEP * grad / (PRESSURE);

		if (y > 0)
		{
			// x valid
			values.y -= velocity.x;
		}

		if (x > 0)
		{
			// y valid
			values.z -= velocity.y;
		}
		surf2Dwrite<float4>(values, outputSurfRef, x * sizeof(float4), y);

	}
}

void __global__ transformTest(int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 sampler1 = sl(x + 1, y + 1, width, height);
		float2 sampler2 = sl(x + 1, y - 1, width, height);
		float2 sampler3 = sl(x - 1, y - 1, width, height);
		float2 sampler4 = sl(x - 1, y + 1, width, height);
		float4 values = tex2D(texRef, sampler1.x, sampler1.y) + tex2D(texRef, sampler2.x, sampler2.y) + tex2D(texRef, sampler3.x, sampler3.y) + tex2D(texRef, sampler4.x, sampler4.y);
		surf2Dwrite<float4>(values / 4, outputSurfRef, x * sizeof(float4), y);
	}
}

__global__ void writeBack(int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float4 value = surf2Dread<float4>(outputSurfRef, x * sizeof(float4), y);
		surf2Dwrite<float4>(value, inputSurfRef, x * sizeof(float4), y);
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
	float4* h_data = new float4[width * height];
	size_t size = width * height * sizeof(float4);
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
	std::fill_n(h_data, width * height, make_float4(0.0f));
	h_data[width*(height / 2) + width / 4 + 1].x = 5;
	h_data[width*(height / 2 + 1) + width / 4 + 1].x =5;
	h_data[width*(height / 2 + 1) + width / 4].x = 5;
	h_data[width*(height / 2) + width / 4].x = 5;
	
	// Allocate CUDA arrays in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	cudaArray* cuInputArray;
	CUDA(cudaMallocArray(&cuInputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

	cudaArray* cuOutputArray;
	CUDA(cudaMallocArray(&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

	// Copy to device memory some data located at address h_data
	// in host memory 
	CUDA(cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size, cudaMemcpyHostToDevice));

	// Bind the arrays to the surface references
	CUDA(cudaBindSurfaceToArray(inputSurfRef, cuInputArray));
	CUDA(cudaBindSurfaceToArray(outputSurfRef, cuOutputArray));

	texRef.normalized = true;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.addressMode[0] = cudaAddressModeClamp; // extend at border
	texRef.addressMode[1] = cudaAddressModeClamp;
	texRef.addressMode[2] = cudaAddressModeClamp;

	CUDA(cudaBindTextureToArray(&texRef, cuInputArray, &channelDesc));

	// Invoke kernel
	dim3 dimBlock(32, 32);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	firstVelocityIteration << <dimGrid, dimBlock >> > (width, height);
	CUDA(cudaDeviceSynchronize());
	writeBack << <dimGrid, dimBlock >> > (width, height);
	CUDA(cudaDeviceSynchronize());

	for (int i = 0; i < ITERATIONS; i++)
	{
		iteratePressure<< <dimGrid, dimBlock >> > (width, height);
		CUDA(cudaDeviceSynchronize());
		writeBack << <dimGrid, dimBlock >> > (width, height);
		CUDA(cudaDeviceSynchronize());

		iterateVelocity<< <dimGrid, dimBlock >> > (width, height);
		CUDA(cudaDeviceSynchronize());
		writeBack << <dimGrid, dimBlock >> > (width, height);
		CUDA(cudaDeviceSynchronize());
	}
	

	CUDA(cudaPeekAtLastError());

	CUDA(cudaMemcpyFromArray(h_data, cuOutputArray, 0, 0, size, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDA(cudaFreeArray(cuInputArray));
	CUDA(cudaFreeArray(cuOutputArray));

	writeOutput(h_data, width, height, "output.ppm");
	
	delete[] h_data;

	return 0;
}

void writeOutput(float4* data, int width, int height, const std::string& path)
{
	float* output = new float[width * height * 3];

	for (int i = 0; i < width * height; i++)
	{
		float value = clamp(data[i].x * 0.5f + 0.5f, 0.0f, 1.0f);
		output[3 * i + 0] = value;
		output[3 * i + 1] = value;
		output[3 * i + 2] = value; //sqrt(1 - value1 * value1 - value2 * value2);
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