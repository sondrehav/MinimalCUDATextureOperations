#include "textureRenderer.h"
#include "debug.h"
#include <glad/glad.h>
#include "shader.h"

// todo: fix a file...
static const std::string vertSource = "#version 330 core\n\nlayout(location = 0) in vec2 position;\nlayout(location = 1) in vec2 uvcoord;\n\nout vec2 uv;\n\nvoid main()\n{\n	uv = uvcoord;\n    gl_Position = vec4(2 * position, 0.0, 1.0);\n}";
static const std::string fragSource = "#version 330 core\n\nin vec2 uv;\nuniform sampler2D tex;\nuniform sampler1D lut;\n\nvoid main()\n{\n	float value = 1.0 / (1.0 + exp(texture(tex, uv).r));\n	gl_FragColor = texture(lut, value);\n}";

TextureRenderer::TextureRenderer()
{
	GL(glGenBuffers(1, &vertexBufferId));
	GL(glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));
	GL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), (GLvoid*)vertexData, GL_STATIC_DRAW));

	GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(0)));
	GL(glEnableVertexAttribArray(0));

	GL(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void*)(sizeof(float) * 3)));
	GL(glEnableVertexAttribArray(1));

	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

	shader = std::make_unique<Shader>();

	GL(shader->attach(vertSource, GL_VERTEX_SHADER));
	GL(shader->attach(fragSource, GL_FRAGMENT_SHADER));

	GL(shader->link());
	GL(shader->use());
	
	GL(glUniform1i(shader->getUniformLocation("tex"), 0));
	GL(glUniform1i(shader->getUniformLocation("lut"), 1));

	GL(shader->validate());

	GL(glGenTextures(1, &lutTextureId));
	GL(glBindTexture(GL_TEXTURE_1D, lutTextureId));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	GL(glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 5, 0, GL_RGBA, GL_UNSIGNED_BYTE, lut));
}


void TextureRenderer::render(const Texture* texture)
{
	shader->use();

	GL(glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId));

	GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (GLvoid*)(0)));
	GL(glEnableVertexAttribArray(0));

	GL(glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (GLvoid*)(sizeof(float) * 3)));
	GL(glEnableVertexAttribArray(1));

	GL(glActiveTexture(GL_TEXTURE0));
	GL(glBindTexture(GL_TEXTURE_2D, texture->getID()));

	GL(glActiveTexture(GL_TEXTURE1));
	GL(glBindTexture(GL_TEXTURE_1D, lutTextureId));

	GL(glDrawArrays(GL_TRIANGLES, 0, 6));

	GL(glDisableVertexAttribArray(0));
	GL(glDisableVertexAttribArray(1));
	GL(glBindBuffer(GL_ARRAY_BUFFER, 0));
}
