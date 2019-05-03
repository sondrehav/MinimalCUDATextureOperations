#pragma once
#include <string>

inline std::string getErrorString(int code) {
	switch (code) {
	case 0x0000: return "GL_NO_ERROR";
	case 0x0500: return "GL_INVALID_ENUM";
	case 0x0501: return "GL_INVALID_VALUE";
	case 0x0502: return "GL_INVALID_OPERATION";
	case 0x0503: return "GL_STACK_OVERFLOW";
	case 0x0504: return "GL_STACK_UNDERFLOW";
	case 0x0505: return "GL_OUT_OF_MEMORY";
	case 0x0506: return "GL_INVALID_FRAMEBUFFER_OPERATION";
	case 0x0507: return "GL_CONTEXT_LOST";
	}
	return std::to_string(code);
}

#define GL(x) x; { int err = glGetError(); if (err != 0) {\
		std::string s = "OpenGL error!\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\tfunc: ";\
		s += #x;\
		s += "\n\tcode: ";\
		s += getErrorString(err).c_str();\
		s += '\n';\
		throw std::exception(s.c_str()); \
	}}

#define ASSERT(x) {bool v = x; if(!x) {\
		std::string s = "Assertion error!\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\tfunc: ";\
		s += #x;\
		s += '\n';\
		throw std::exception(s.c_str()); \
	}} 

#define ERROR(x, type)  {\
		std::string s = type; \
		s += " error!\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\tlog: \n\n";\
		s += #x;\
		s += '\n\n';\
		throw std::exception(s.c_str()); \
	}