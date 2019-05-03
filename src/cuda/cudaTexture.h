#pragma once
#include "../gl/texture.h"
#include <cuda.h>
#include <cudaGL.h>
#include "../cudaDebug.h"

class RWCUDATexture2D : public Texture
{
public:
	RWCUDATexture2D(int width, int height, GLenum internalFormat);

	void setData(void* data, GLenum dataFormat, GLenum dataType) override;

	void bindToTextureRef(CUmodule module, CUtexref texReadRef);
	void bindToSurfaceRef(CUmodule module, CUsurfref surfWriteRef);
	
	const int width;
	const int height;

private:

	CUarray cudaArray;
	CUgraphicsResource cudaGraphicsResource;


};
