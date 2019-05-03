#pragma once

#include <glad/glad.h>
#include <string>
#include <map>


class Shader
{
	
public:

	Shader();
	~Shader();

	bool attach(const std::string& source, GLenum type);
	bool link();
	void use();
	bool validate();
	bool isValid() const { return programID > 0 && valid; }

	int getUniformLocation(const std::string& name);

	static void deactivateCurrentShader();

	unsigned int getProgramID() { return programID; }

private:
	unsigned int programID = 0;
	std::map<std::string, GLint> uniformLocations;
	bool valid = false;


};
