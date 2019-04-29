#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <algorithm>
#include <string>

#include "helper_math.h"
#include "noise.h"

#define ITERATIONS 2

void ppm(float* data, int width, int height, const std::string& path);
void writeOutput(float4* data, int width, int height, const std::string& path);

// static __device__ __forceinline__ void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap)
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;

// --- 2D float4 texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;

__device__ float2 getSamplerLocation(float2 location, int2 dimension)
{
	return make_float2(((float)location.x + 0.5f) / (float)dimension.x, ((float)location.y + 0.5f) / (float)dimension.y);
}

// Simple copy kernel
__global__ void iterate(int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		float2 samplerLocation = getSamplerLocation(make_float2(x, y), make_int2(width, height));
		float2 samplerLocation1 = getSamplerLocation(make_float2(x - 1, y - 1), make_int2(width, height));
		float2 samplerLocation2 = getSamplerLocation(make_float2(x - 1, y + 1), make_int2(width, height));
		float2 samplerLocation3 = getSamplerLocation(make_float2(x + 1, y + 1), make_int2(width, height));
		float2 samplerLocation4 = getSamplerLocation(make_float2(x + 1, y - 1), make_int2(width, height));

		float4 value = 5 * tex2D(texRef, samplerLocation.x, samplerLocation.y);
		value -= tex2D(texRef, samplerLocation1.x, samplerLocation1.y);
		value -= tex2D(texRef, samplerLocation2.x, samplerLocation2.y);
		value -= tex2D(texRef, samplerLocation3.x, samplerLocation3.y);
		value -= tex2D(texRef, samplerLocation4.x, samplerLocation4.y);
		
		// Write to output surface
		surf2Dwrite<float4>(value, outputSurfRef, x * sizeof(float4), y);
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

	const size_t width = 512, height = 512;
	float4* h_data = new float4[width * height];
	size_t size = width * height * sizeof(float4);


	float4 min = make_float4(std::numeric_limits<float>::max());
	float4 max = make_float4(std::numeric_limits<float>::min());
	for(int i = 0; i < height; i++)
	for(int j = 0; j < width; j++)
	{
		float4 value = make_float4(
			perlin2d((float)j / width + 0000, (float)i / height + 000, 4.0, 12),
			perlin2d((float)j / width + 8734, (float)i / height + 834, 4.0, 12),
			perlin2d((float)j / width + 1637, (float)i / height + 137, 4.0, 12),
			0.0f
		);
		h_data[i * width + j] = value;
		min = fminf(min, value);
		max = fmaxf(max, value);
	}
	for (int i = 0; i < height; i++)
	for (int j = 0; j < width; j++)
	{
		h_data[i * width + j] = 0.5 * (h_data[i * width + j] - min) / (max - min) + 0.25;
	}
	
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
	texRef.addressMode[0] = cudaAddressModeMirror; // extend at border
	texRef.addressMode[1] = cudaAddressModeMirror;
	texRef.addressMode[2] = cudaAddressModeMirror;

	CUDA(cudaBindTextureToArray(&texRef, cuInputArray, &channelDesc));

	// Invoke kernel
	dim3 dimBlock(32, 16);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

	for (int i = 0; i < ITERATIONS; i++)
	{
		iterate << <dimGrid, dimBlock >> > (width, height);
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
		output[3 * i + 0] = data[i].x;
		output[3 * i + 1] = data[i].y;
		output[3 * i + 2] = data[i].z;
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
			color[0] = (unsigned char) (data[3 * (j * width + i) + 0] * 256);
			color[1] = (unsigned char) (data[3 * (j * width + i) + 1] * 256);
			color[2] = (unsigned char) (data[3 * (j * width + i) + 2] * 256);
			(void)fwrite(color, 1, 3, fp);
		}
	}
	(void)fclose(fp);
}