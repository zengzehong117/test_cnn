#ifndef INNERPRODUCT_CPP
#define INNERPRODUCT_CPP

#include "innerproduct_layer.hpp"

namespace Test_cnn{
	void Innerproduct_layer::forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){
		float *in_data = bottom_blobs[0]->get_data();
		float *out_data = top_blobs[0]->creat_get_data_ptr();
		for (int i = 0; i < top_blobs[0]->ge_lenth(); i++){
			*(out_data + i) = 0;
		}
		float *multi = multiplier_->get_data();

		float *bias_multi = bias_multiplier_->get_data();

		int conv_out_channels_ = param_->innerproduct_param().num_output();
		
		int kernel_dim_ = bottom_blobs[0]->ge_lenth();
		for (int c = 0; c < conv_out_channels_; c++){
			
				for (int e = 0; e < kernel_dim_; e++){
					
					//ccc = *(col_buff + d + e*conv_out_spatial_dim_) * (*(weights + e + c*kernel_dim_));
					//*(output + d + c*conv_out_spatial_dim_) += 0.25*ccc; // 0.08*(0.17 - ccc*ccc);
					//*(output + d + c*conv_out_spatial_dim_) += 0.08*(0.4- (*(col_buff + d + e*conv_out_spatial_dim_) - (*(weights + e + c*kernel_dim_)))*(*(col_buff + d + e*conv_out_spatial_dim_) - (*(weights + e + c*kernel_dim_))));
					*(out_data + c) += *(in_data + e)*(*(multi + e+c*kernel_dim_));
				}
				*(out_data + c) += *(bias_multi + c);
			
		}




	}




}






#endif // !INNERPRODUCT_CPP
