#pragma once

#include <string>
#include <iomanip>
#include <ctime>
#include <sstream>

inline std::string timestamp()
{
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
	return oss.str();
}

/*
 * Output a debug string.
 */
#define DEBUG_STR(x) { \
		std::string s = "Debug message:\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\ttime: ";\
		s += timestamp();\
		s += "\n\tmessage: ";\
		s += x;\
		s += '\n';\
		printf("%s\n", s.c_str());\
		}
 /*
  * Output a debug string.
  */
#define DEBUG_STR(a, b) { \
		std::string s = "Debug message:\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\ttime: ";\
		s += timestamp();\
		s += "\n\tmessage: ";\
		s += std::string(a) + std::string(b);\
		s += '\n';\
		printf("%s\n", s.c_str());\
		}
