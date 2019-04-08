#ifndef POOLING_LAYER_CPP
#define POOLING_LAYER_CPP

#include "pooling_layer.hpp"

namespace Test_cnn{
	void Pooling_layer::init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){
		vector<int> top_shape;
		PoolingParameter param = param_->pooling_param();
		top_shape.push_back(bottom_blobs[0]->get_shape_0());
		top_shape.push_back(bottom_blobs[0]->get_shape_1());
		//static_cast<int>(ceil(static_cast<float>(		height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
		int shape2 = static_cast<int>(static_cast<float>(bottom_blobs[0]->get_shape_2() - param.kernel()) / param.stride() + 1);
		top_shape.push_back(shape2);
		top_shape.push_back(shape2);
		top_blobs[0]->set_shape(top_shape);
		cout << "pool shape:" << top_shape[2] << endl;
	}
	void Pooling_layer::forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){
		float*in_data = bottom_blobs[0]->get_data();
		float*out_data = top_blobs[0]->creat_get_data_ptr();
		
		PoolingParameter param = param_->pooling_param();
		/*
		for (int i = 0; i < top_blobs[0]->get_shape_2(); i++)
		{
			for (int j = 0; j < top_blobs[0]->get_shape_2(); j++)
			{
				out_data[i];311
				for (int j = 0; j < param.kernel(); j++){

				}
			}
		}
		*/
		if (param.pool() == "ave"){
			for (int i = 0; i < top_blobs[0]->ge_lenth(); i++)
			{
				out_data[i] = 0;
			}
			for (int n = 0; n < bottom_blobs[0]->get_shape_0(); ++n) {
				for (int c = 0; c < bottom_blobs[0]->get_shape_1(); ++c) {
					for (int ph = 0; ph < top_blobs[0]->get_shape_2(); ++ph) {
						for (int pw = 0; pw < top_blobs[0]->get_shape_3(); ++pw) {
							int hstart = ph * param.stride();
							int wstart = pw *  param.stride();
							int hend = std::min(hstart + param.kernel(), bottom_blobs[0]->get_shape_2());
							int wend = std::min(wstart + param.kernel(), bottom_blobs[0]->get_shape_3());
							hstart = max(hstart, 0);
							wstart = max(wstart, 0);
							const int pool_index = c*top_blobs[0]->get_shape_2()*top_blobs[0]->get_shape_2() + ph * top_blobs[0]->get_shape_2() + pw;
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									const int index = c*bottom_blobs[0]->get_shape_3()*bottom_blobs[0]->get_shape_3() + h * bottom_blobs[0]->get_shape_3() + w;
									
									out_data[pool_index] += in_data[index];
									
								}
							}
							out_data[pool_index] /= param.kernel()*param.kernel();
						}
					}
				}
			}
		}
		else if (param.pool() == "max"){
			for (int n = 0; n < bottom_blobs[0]->get_shape_0(); ++n) {
				for (int c = 0; c < bottom_blobs[0]->get_shape_1(); ++c) {
					for (int ph = 0; ph < top_blobs[0]->get_shape_2(); ++ph) {
						for (int pw = 0; pw < top_blobs[0]->get_shape_3(); ++pw) {
							int hstart = ph * param.stride();
							int wstart = pw *  param.stride();
							int hend = std::min(hstart + param.kernel(), bottom_blobs[0]->get_shape_2());
							int wend = std::min(wstart + param.kernel(), bottom_blobs[0]->get_shape_3());
							hstart = max(hstart, 0);
							wstart = max(wstart, 0);
							const int pool_index = c*top_blobs[0]->get_shape_2()*top_blobs[0]->get_shape_2()+ph * top_blobs[0]->get_shape_2() + pw;
							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									const int index = c*bottom_blobs[0]->get_shape_3()*bottom_blobs[0]->get_shape_3()+h * bottom_blobs[0]->get_shape_3() + w;
									if (in_data[index] > out_data[pool_index]) {
										out_data[pool_index] = in_data[index];
									}
								}
							}
						}
					}
				}
			}
		}
		else{
			cout << "unkown pool type." << endl;
		}
	}
}



#endif // !POOLING_LAYER_CPP
