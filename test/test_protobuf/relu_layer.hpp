#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
#include "layer.h"


using namespace std;
namespace Test_cnn{
	class Relu_layer : public Test_layer
	{
	public:
		Relu_layer(){}
		Relu_layer(const LayerParameter &_param){
			param_ = &_param;
			//multiplier_.reset(new Blob());
			cout << "creat Relu layer and init param." << endl;
		}
		~Relu_layer(){}

		void forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){

			float *top_data = top_blobs[0]->get_data();
			float*bottom_data = bottom_blobs[0]->get_data();
			for (int i = 0; i < top_blobs[0]->ge_lenth(); i++)
			{
				top_data[i] = std::max(bottom_data[i],static_cast<float> (0));
			}
		}


	private:

	};


}


#endif