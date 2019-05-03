#include "shader.h"
#include <cassert>
#include "debug.h"


Shader::Shader()
{
	programID = GL(glCreateProgram());
}

Shader::~Shader()
{
	if(programID > 0)
	{
		GL(glDeleteProgram(programID));
	}
}

bool Shader::attach(const std::string& source, GLenum type)
{
	ASSERT(programID > 0);

	int shader = GL(glCreateShader(type));
	const char* s = source.c_str();
	GL(glShaderSource(shader, 1, &s, nullptr));
	GL(glCompileShader(shader));

	int compileStatus;
	GL(glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus));
	if(compileStatus == 0)
	{
		int logLength;
		GL(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength));
		char* log = new char[logLength];
		GL(glGetShaderInfoLog(shader, logLength, nullptr, log));
		std::string error(log);
		delete[] log;
		ERROR(error, "VALIDATION");
	}

	GL(glAttachShader(programID, shader));
	GL(glDeleteShader(shader));
	return true;
}

bool Shader::link()
{
	ASSERT(programID > 0);

	GL(glLinkProgram(programID));

	int linkStatus;
	GL(glGetProgramiv(programID, GL_LINK_STATUS, &linkStatus));
	if(linkStatus == 0)
	{
		int logLength;
		GL(glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &logLength));
		char* log = new char[logLength];
		GL(glGetProgramInfoLog(programID, logLength, nullptr, log));
		std::string error(log);
		delete[] log;
		ERROR(error, "PROGRAM");
	}
	return true;
	
}

void Shader::use()
{
	GL(glUseProgram(programID));
}

bool Shader::validate()
{

	GL(glValidateProgram(programID));

	int validateStatus;
	GL(glGetProgramiv(programID, GL_VALIDATE_STATUS, &validateStatus));
	if (validateStatus == 0)
	{
		int logLength;
		GL(glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &logLength));
		char* log = new char[logLength];
		GL(glGetProgramInfoLog(programID, logLength, nullptr, log));
		std::string error(log);
		delete[] log;
		ERROR(error, "VALIDATION");
	}
	valid = true;
	return true;
}

void Shader::deactivateCurrentShader()
{
	GL(glUseProgram(0));
}

int Shader::getUniformLocation(const std::string& name)
{
	if(uniformLocations.find(name) != uniformLocations.end())
	{
		return uniformLocations[name];
	}
	GLint location = GL(glGetUniformLocation(programID, name.c_str()));
	
	if(location < 0)
	{
		ERROR(std::string("Could not find uniform location: ") + name, "Uniform not found")
	}
	uniformLocations[name] = location;
	return location;
}
