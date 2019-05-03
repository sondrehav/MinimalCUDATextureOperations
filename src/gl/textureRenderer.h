#pragma once
#include "shader.h"
#include "texture.h"

class TextureRenderer
{

public:
	TextureRenderer();
	void render(const Texture*);

private:
	std::unique_ptr<Shader> shader;

	const float vertexData[5 * 6] = {
		 0.5f,  0.5f, 0.0f, 1.0f, 0.0f,  // top right
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  // bottom right
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		-0.5f, -0.5f, 0.0f, 0.0f, 1.0f,  // bottom left
		-0.5f,  0.5f, 0.0f, 0.0f, 0.0f,  // top left 
		 0.5f, -0.5f, 0.0f, 1.0f, 1.0f  // bottom right
	};

	const uint32_t lut[5] = {
		0xffffffff,
		0xffbbbbbb,
		0xff888888,
		0xff444444,
		0xff000000
	};

	GLuint vertexBufferId;
	GLuint lutTextureId;

};
