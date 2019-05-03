#pragma once

#define GLM_FORCE_PURE

#include <glm/glm.hpp>
#include <vector>

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct Mesh
{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	glm::vec3 minExtent;
	glm::vec3 maxExtent;

	void calculateExtents()
	{

		this->minExtent = glm::vec3(std::numeric_limits<float>::max());
		this->maxExtent = glm::vec3(std::numeric_limits<float>::min());

		for(auto vertex : vertices)
		{
			this->minExtent = glm::min(vertex.position, this->minExtent);
			this->maxExtent = glm::max(vertex.position, this->maxExtent);
		}
	}

};