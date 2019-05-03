#pragma once

#include "mesh.h"
#include <glad/glad.h>
#include "debug.h"

namespace globjects {
	class Shader;
}

class MeshWrapper {

public:
	MeshWrapper(const Mesh* mesh) : mesh(mesh)
	{

		GL(glGenVertexArrays(1, &vao));

		GL(glGenBuffers(1, &vbo));
		GL(glGenBuffers(1, &ibo));

		// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
		GL(glBindVertexArray(vao));

		GL(glBindBuffer(GL_ARRAY_BUFFER, vbo));
		GL(glBufferData(GL_ARRAY_BUFFER, mesh->vertices.size() * sizeof(Vertex), mesh->vertices.data(), GL_STATIC_DRAW));

		GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo));
		GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->indices.size() * sizeof(uint32_t), mesh->indices.data(), GL_STATIC_DRAW));

		GL(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position)));
		GL(glEnableVertexAttribArray(0));

		GL(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal)));
		GL(glEnableVertexAttribArray(1));

		GL(glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv)));
		GL(glEnableVertexAttribArray(2));

		// note that this is allowed, the call to glVertexAttribPointer registered vbo as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
		GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

		// remember: do NOT unbind the ibo while a vao is active as the bound element buffer object IS stored in the vao; keep the ibo bound.
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// You can unbind the vao afterwards so other vao calls won't accidentally modify this vao, but this rarely happens. Modifying other
		// vaos requires a call to glBindVertexArray anyways so we generally don't unbind vaos (nor vbos) when it's not directly necessary.
		GL(glBindVertexArray(0));
	}

	~MeshWrapper()
	{
		GL(glBindVertexArray(vao));
		GL(glDeleteBuffers(1, &vbo));
		GL(glDeleteBuffers(1, &ibo));
		GL(glBindVertexArray(0));
		GL(glDeleteVertexArrays(1, &vao));
		mesh = nullptr;
	}

	void draw(globjects::Shader& shader)
	{
		GL(glBindVertexArray(vao));
		GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo));

		GL(glDrawElements(GL_TRIANGLES, mesh->indices.size(), GL_UNSIGNED_INT, 0));

		GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
		GL(glBindVertexArray(0));
		
	}

private:
	const Mesh* mesh;
	unsigned int vbo, vao, ibo;

};
