// Surfaces for writing
#include <cstdio>
#include <driver_types.h>
#include <texture_types.h>

//surface<void, 2> velocitySurfRef;
surface<void, 2> pressureSurfRef;

// Textures for reading
//texture<float2, 2, cudaReadModeElementType> velocityTexRef;
texture<float, 2, cudaReadModeElementType> pressureTexRef;

extern "C" __global__ void sharpenFunction(int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < width && y < height)
	{
		
		float fx = x + 0.5;
		float fy = y + 0.5;

		float valTop = tex2D(pressureTexRef, fx, fy - 1);
		float valBottom = tex2D(pressureTexRef, fx, fy + 1);
		float valLeft = tex2D(pressureTexRef, fx - 1, fy);
		float valRight = tex2D(pressureTexRef, fx + 1, fy);
		float val = tex2D(pressureTexRef, fx, fy);

		float newVal = (valTop + valBottom + valLeft + valRight + 4 * val) / 8.0f;
		 
		surf2Dwrite<float>(newVal, pressureSurfRef, x * sizeof(float), y);
		
	}
}
