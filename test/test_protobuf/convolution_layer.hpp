#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#define USE_CUDA 0


#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
#include "layer.h"


extern"C"
void set_cuda_data(int _lenth, const float *cpu_addr, float *gpu_addr);
extern"C"
void get_cuda_data(int _lenth, float *cpu_addr, const float *gpu_addr);


extern"C"
void forward_conv_gpu(const float *_input, float *_im2col,
float*_kenel_data, float *_out,
const int _in_with1, const int _kenel_num, const int _out_high,
const int _in_channels, const int _in_height, const int _in_width,
const int _kernel_h, const int _kernel_w,
const int pad, const int stride);

using namespace std;


namespace Test_cnn{
	class Convolution_layer : public Test_layer
	{
	public:
		Convolution_layer(){
			cout << "creat convolution layer." << endl;
		}
		Convolution_layer(const LayerParameter &_param){
			param_ = &_param;
			multiplier_.reset(new Blob());
			cout << "creat convolution layer and init param,blob." << endl;
		}
		~Convolution_layer(){}

		void init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);
		void init_multiplier(const Myblob &_param){
			multiplier_->set_blob_from_model(_param);
		}

		void forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);
		bool is_a_ge_zero_and_a_lt_b(int a, int b);
		void im2col_cpu(const float* data_im, const int channels,
			const int height, const int width, const int kernel_h, const int kernel_w,
			const int pad_h, const int pad_w,
			const int stride_h, const int stride_w,
			const int dilation_h, const int dilation_w,
			float* data_col);



	private:
		void change_inblob(const float *bottom);
		Blob changed_blob;

	};


}


#endif