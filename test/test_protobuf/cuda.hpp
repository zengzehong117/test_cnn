#ifndef CU_HPP_
#define CU_HPP_

#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<device_launch_parameters.h>
#include "cuda.hpp"
#include "cublas_v2.h"

extern"C"
float *get_cuda_ptr(int _lenth){
	float *ptr;
	cudaMalloc((void**)&ptr,
		_lenth* sizeof(float));
	return ptr;

}

extern"C"
void set_cuda_data(int _lenth, const float *cpu_addr, float *gpu_addr){
	cublasSetVector(
		         _lenth,    // Ҫ�����Դ��Ԫ�ظ���
		         sizeof(float),    // ÿ��Ԫ�ش�С
				 cpu_addr,  // ��������ʼ��ַ
		         1,    // ����Ԫ��֮��Ĵ洢���
				 gpu_addr,    // GPU ����ʼ��ַ
		         1    // ����Ԫ��֮��Ĵ洢���
		);
}

extern"C"
void get_cuda_data(int _lenth, float *cpu_addr, const float *gpu_addr){
	cublasGetVector(
		_lenth,    //  Ҫȡ��Ԫ�صĸ���
		         sizeof(float),    // ÿ��Ԫ�ش�С
				 gpu_addr,    // GPU ����ʼ��ַ
		         1,    // ����Ԫ��֮��Ĵ洢���
				 cpu_addr,    // ��������ʼ��ַ
		         1    // ����Ԫ��֮��Ĵ洢���
		);
}


#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace Test_cnn {

	extern"C"
	void forward_conv_gpu(const float *_input, float *_im2col,
		float*_kenel_data, float *_out,
		const int _in_with1, const int _kenel_num, const int _out_high,
		const int _in_channels, const int _in_height, const int _in_width,
		const int _kernel_h, const int _kernel_w,
		const int pad, const int stride);


	
	void im2col_gpu(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		float* data_col);


	void im2col_bool_gpu(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, const int dilation_h, const int dilation_w,
		float* data_col);



}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
