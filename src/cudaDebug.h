#pragma once

#include <string>

/*
 * Checks CUDA runtime errors and aborts if something is wrong.
 */
#define CUDA(func) { cudaError_t err = func; if (err != cudaSuccess) { \
		std::string s = "CUDA error!\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\tfunc: ";\
		s += #func;\
		s += "\n\tcode: ";\
		s += cudaGetErrorName(err);\
		s += '\n';\
		throw std::exception(s.c_str()); }}\


/*
 * Checks CUDA driver API errors and aborts if something is wrong.
 */
#define CUDA_D(func) { CUresult err = func; if (err	!= CUDA_SUCCESS) { \
		std::string s = "CUDA error!\n\tfile: ";\
		s += __FILE__;\
		s += "\n\tline: ";\
		s += std::to_string(__LINE__);\
		s += "\n\tfunc: ";\
		s += #func;\
		s += "\n\tcode: ";\
		s += std::to_string(err);\
		s += '\n';\
		throw std::exception(s.c_str()); }}\

