#include "helper_math.h"
#include "integrator.h"



float __inline__ __device__ amplitude(float distanceFromCenter, float brushSize, float brushFalloff)
{
	float f = (-abs(distanceFromCenter) + brushSize) / brushFalloff + 1;
	return fmaxf(fminf(f, 1.0f), 0.0f);
}

void __global__ drawLine(int2 from, int2 to, float brushSize, float brushFalloff, int width, int height)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		int2 p = make_int2(x, y);
		float2 fp = make_float2(p);
		float2 ffrom = make_float2(from);
		float2 fto = make_float2(to);

		float2 proj = fto - ffrom;
		float2 vproj = dot(fp - ffrom, proj) / dot(proj, proj) * proj;

		float vLength = length(fp - vproj);

		float toLength = length(fp - fto);
		float fromLength = length(fp - ffrom);

		float amp = 0;

		if (dot(fp - ffrom, vproj) < 0.0f && fromLength < brushSize + brushFalloff)
		{
			// is part of the circle at the bottom edge... (from)
			amp = amplitude(fromLength, brushSize, brushFalloff);

		}
		else if (dot(fp - fto, -vproj) < 0.0f && toLength < brushSize + brushFalloff)
		{
			// is part of the circle at the top edge... (to)
			amp = amplitude(toLength, brushSize, brushFalloff);
		}
		else if (vLength < brushSize + brushFalloff)
		{
			// is part of line ...
			amp = amplitude(vLength, brushSize, brushFalloff);
		}

		surf2Dwrite<float>(amp, geometrySurface, x * sizeof(float), y);
	}
}


void Integrator::drawGeometryLine(int2 from, int2 to, float brushSize, float brushFalloff)
{
	drawLine << <dimGrid, dimBlock >> > (from, to, brushSize, brushFalloff, params.actualWidth, params.actualHeight);
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaPeekAtLastError());
}
