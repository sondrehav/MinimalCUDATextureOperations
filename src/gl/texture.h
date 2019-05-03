#pragma once
#include <glad/glad.h>
#include "debug.h"
#include <cassert>


class Texture
{
public:
	Texture(GLenum internalFormat, GLenum minMagFilter, GLenum textureType) : internalFormat(internalFormat), minMagFilter(minMagFilter), textureType(textureType) {}
	
	~Texture()
	{
		if(textureId > 0)
		{
			glDeleteTextures(1, &textureId);
		}
	}


	virtual void bind(int slot = 0)
	{
		assert(isValid());
		GL(glActiveTexture(GL_TEXTURE0 + slot));
		GL(glBindTexture(textureType, textureId));
	}

	virtual void setData(void* data, GLenum dataFormat, GLenum dataType) = 0;

	bool isValid() const { return textureId > 0; }

	GLuint getID() const { return textureId; }

protected:
	GLenum internalFormat = GL_RGB8;
	GLenum minMagFilter = GL_LINEAR;
	GLenum textureType;
	GLuint textureId = 0;

};


class Texture2D : public Texture
{
public:
	Texture2D(int width, int height, GLenum internalFormat = GL_RGB8, void* data = nullptr, GLenum dataFormat = 0, GLenum dataType = 0) : Texture(internalFormat, GL_LINEAR, GL_TEXTURE_2D), width(width), height(height)
	{
		if(data != nullptr && dataFormat != 0 && dataType != 0)
		{
			this->setDataInternal(data, dataFormat, dataType);
		}
	}

	void setData(void* data, GLenum dataFormat, GLenum dataType) override
	{	
		if (textureId <= 0)
		{
			GL(glGenTextures(1, &textureId));
		}
		GL(glBindTexture(GL_TEXTURE_2D, textureId));
		GL(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, dataFormat, dataType, data));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minMagFilter));
		GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, minMagFilter));
		GL(glBindTexture(GL_TEXTURE_2D, 0));
	}

	const int width;
	const int height;

private:
	void setDataInternal(void* data, GLenum dataFormat, GLenum dataType) { setData(data, dataFormat, dataType); }

};

class Texture3D : public Texture
{
public:
	Texture3D(int width, int height, int depth, GLenum internalFormat = GL_RGB8, void* data = nullptr, GLenum dataFormat = 0, GLenum dataType = 0) : Texture(internalFormat, GL_LINEAR, GL_TEXTURE_3D), width(width), height(height), depth(depth)
	{
		if (data != nullptr && dataFormat != 0 && dataType != 0)
		{
			this->setDataInternal(data, dataFormat, dataType);
		}
	}

	void setData(void* data, GLenum dataFormat, GLenum dataType) override
	{
		if (textureId <= 0)
		{
			GL(glGenTextures(1, &textureId));
		}
		GL(glBindTexture(GL_TEXTURE_3D, textureId));
		GL(glTexImage3D(GL_TEXTURE_3D, 0, internalFormat, width, height, depth, 0, dataFormat, dataType, data));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, minMagFilter));
		GL(glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, minMagFilter));
		GL(glBindTexture(GL_TEXTURE_3D, 0));
	}

	const int width;
	const int height;
	const int depth;

private:
	void setDataInternal(void* data, GLenum dataFormat, GLenum dataType) { setData(data, dataFormat, dataType); }

};

