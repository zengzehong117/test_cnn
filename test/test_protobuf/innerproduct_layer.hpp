#ifndef INNERPRODUCT_LAYER_H
#define INNERPRODUCT_LAYER_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
#include "layer.h"


using namespace std;
namespace Test_cnn{
	class Innerproduct_layer : public Test_layer
	{
	public:
		Innerproduct_layer(){
			cout << "creat innerproduct layer." << endl;
		}
		Innerproduct_layer(const LayerParameter &_param){
			param_ = &_param;
			multiplier_.reset(new Blob());
			bias_multiplier_.reset(new Blob());

			cout << "creat Innerproduct layer and init param." << endl;
		}
		~Innerproduct_layer(){}
		void forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);



		void init_multiplier(const Myblob &_param){
			if (multiplier_->is_null()==NULL){
				multiplier_->set_blob_from_model(_param);
			}
			else
			{
				bias_multiplier_->set_blob_from_model(_param);
			}
		}

		virtual void init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){
			InnerproductParameter param = param_->innerproduct_param();
			vector<int> shape;
			shape.push_back(bottom_blobs[0]->ge_lenth());
			shape.push_back(param.num_output());
			multiplier_->set_shape(shape);
			cout << "Innerproduct shape:" << shape[0] << "  " << shape[1] << endl;

			vector<int> shape1;
			
			shape1.push_back(param.num_output());
			bias_multiplier_->set_shape(shape1);
			cout << "Innerproduct shape1:" << shape1[0] << endl;

			vector<int> top_shape;
			top_shape.push_back(param.num_output());
			top_blobs[0]->set_shape(top_shape);
		}


	private:

	};


}


#endif