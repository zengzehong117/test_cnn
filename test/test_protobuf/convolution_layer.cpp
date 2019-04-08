#ifndef CONVOLUTION_LAYER_CPP
#define CONVOLUTION_LAYER_CPP

#include "convolution_layer.hpp"

namespace Test_cnn{
	void Convolution_layer::init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){
		/*
		int top_demension_0 = 1;
		int top_demension_1 = 1;
		int top_demension_2 = 1;
		int top_demension_3 = 1;
		int multiplier_demension_0 = 1;
		int multiplier_demension_1 = 1;
		int multiplier_demension_2 = 1;
		int multiplier_demension_3 = 1;
		*/
		vector<int> top_shape;
		vector<int> multiplier_shape;
		ConvolutionParameter param = param_->conv_param();
		//multiplier_demension_1 = param->num_output();
		multiplier_shape.push_back(param.num_output());
		multiplier_shape.push_back(bottom_blobs[0]->get_shape_1());
		multiplier_shape.push_back(param.kernel());
		multiplier_shape.push_back(param.kernel());
		multiplier_->set_shape(multiplier_shape);
		//top_shape.push_back(bottom_blobs[0].)
		top_shape.push_back(bottom_blobs[0]->get_shape_0());
		top_shape.push_back(param.num_output());
		int shape2 = bottom_blobs[0]->get_shape_2() + 2 * param.pad() - param.kernel() + 1; //no stride
		top_shape.push_back(shape2);
		top_shape.push_back(shape2);
		top_blobs[0]->set_shape(top_shape);





	}

	void Convolution_layer::forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){
		const float *in_data = bottom_blobs[0]->get_data();
		float *out_data = top_blobs[0]->creat_get_data_ptr();
		for (int i = 0; i < top_blobs[0]->ge_lenth(); i++){
			*(out_data + i) = 0;
		}
		const float *multi = multiplier_->get_data();

		int conv_out_channels_ = param_->conv_param().num_output();
		int conv_out_spatial_dim_ = top_blobs[0]->get_shape_2()*top_blobs[0]->get_shape_2();
		int kernel_dim_ = multiplier_->ge_lenth() / multiplier_->get_shape_0();
		
#if USE_CUDA
		float*in_cuda_data=bottom_blobs[0]->creat_get_cudata_ptr();
		float *cu_multi=multiplier_->creat_get_cudata_ptr();
		float *out_cuda_data = top_blobs[0]->creat_get_cudata_ptr();

		


		set_cuda_data(bottom_blobs[0]->ge_lenth(), in_data, in_cuda_data);
		set_cuda_data(multiplier_->ge_lenth(), multi, cu_multi);
		forward_conv_gpu(in_cuda_data, changed_blob.creat_get_cudata_ptr(),
			cu_multi, out_cuda_data,
			conv_out_spatial_dim_, kernel_dim_, conv_out_channels_,
			top_blobs[0]->get_shape_2(), top_blobs[0]->get_shape_1(), top_blobs[0]->get_shape_0(),
			multiplier_->get_shape_1(), multiplier_->get_shape_0(),
			param_->conv_param().pad(), param_->conv_param().stride()

			);
		get_cuda_data(top_blobs[0]->ge_lenth(), out_data, out_cuda_data);


		//float *out_cuda_data = top_blobs[0]->creat_get_cudata_ptr();
		//float *multi_cu=multiplier_
#endif
		///init changed_blob shape;
		vector<int> changed_shape;
		changed_shape.push_back(bottom_blobs[0]->get_shape_0());
		//changed_shape.push_back(multiplier_->ge_lenth()/multiplier_->get_shape_0());
		//changed_shape.push_back(top_blobs[0]->get_shape_1());
		changed_shape.push_back(top_blobs[0]->get_shape_2()*top_blobs[0]->get_shape_2());
		changed_shape.push_back(multiplier_->ge_lenth() / multiplier_->get_shape_0());
		changed_blob.set_shape(changed_shape);

		im2col_cpu(in_data, bottom_blobs[0]->get_shape_1(),
			bottom_blobs[0]->get_shape_2(), bottom_blobs[0]->get_shape_3(),
			multiplier_->get_shape_2(), multiplier_->get_shape_3(),
			param_->conv_param().pad(), param_->conv_param().pad(),
			param_->conv_param().stride(), param_->conv_param().stride(),1,1,
			changed_blob.creat_get_data_ptr()
			);

		/*int conv_out_channels_ = param_->conv_param().num_output();
		int conv_out_spatial_dim_ = top_blobs[0]->get_shape_2()*top_blobs[0]->get_shape_2();
		int kernel_dim_ = multiplier_->ge_lenth() / multiplier_->get_shape_0();*/
		for (int c = 0; c < conv_out_channels_; c++){
			for (int d = 0; d < conv_out_spatial_dim_; d++){
				for (int e = 0; e < kernel_dim_; e++){
					if (*(multi + e + c*kernel_dim_) == 0){
						continue;
					}
					//ccc = *(col_buff + d + e*conv_out_spatial_dim_) * (*(weights + e + c*kernel_dim_));
					//*(output + d + c*conv_out_spatial_dim_) += 0.25*ccc; // 0.08*(0.17 - ccc*ccc);
					//*(output + d + c*conv_out_spatial_dim_) += 0.08*(0.4- (*(col_buff + d + e*conv_out_spatial_dim_) - (*(weights + e + c*kernel_dim_)))*(*(col_buff + d + e*conv_out_spatial_dim_) - (*(weights + e + c*kernel_dim_))));
					*(out_data + d + c*conv_out_spatial_dim_) += 
						*(changed_blob.get_data() + d + e*conv_out_spatial_dim_)
						*(*(multi + e + c*kernel_dim_));
				}
			}
		}

		/*
		for (int c = 0; c < conv_out_channels_; c++){
			for (int d = 0; d < conv_out_spatial_dim_; d++){
				for (int e = 0; e < kernel_dim_; e++){
					if (*(weights + e + c*kernel_dim_) == 0){
						continue;
					}
					ccc = *(col_buff + d + e*conv_out_spatial_dim_) * (*(weights + e + c*kernel_dim_));
					*(output + d + c*conv_out_spatial_dim_) += 0.25*ccc; // 0.08*(0.17 - ccc*ccc);
					//*(output + d + c*conv_out_spatial_dim_) += 0.08*(0.4- (*(col_buff + d + e*conv_out_spatial_dim_) - (*(weights + e + c*kernel_dim_)))*(*(col_buff + d + e*conv_out_spatial_dim_) - (*(weights + e + c*kernel_dim_))));
					// *(output + d + c*conv_out_spatial_dim_) += *(col_buff + d + e*conv_out_spatial_dim_)*(*(weights + e + c*kernel_dim_));
				}
			}
		}
		*/



	}

	bool Convolution_layer::is_a_ge_zero_and_a_lt_b(int a, int b) {
		return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}

	void Convolution_layer::im2col_cpu(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		float* data_col) {
		const int output_h = (height + 2 * pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int channel_size = height * width;
		for (int channel = channels; channel--; data_im += channel_size) {
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					int input_row = -pad_h + kernel_row * dilation_h;
					for (int output_rows = output_h; output_rows; output_rows--) {
						if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
							for (int output_cols = output_w; output_cols; output_cols--) {
								*(data_col++) = 0;
							}
						}
						else {
							int input_col = -pad_w + kernel_col * dilation_w;
							for (int output_col = output_w; output_col; output_col--) {
								if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
									*(data_col++) = data_im[input_row * width + input_col];
								}
								else {
									*(data_col++) = 0;
								}
								input_col += stride_w;
							}
						}
						input_row += stride_h;
					}
				}
			}
		}
	}



}





#endif // !CONVOLUTION_LAYER_CPP
