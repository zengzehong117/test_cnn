#ifndef GET_LAYER_HPP
#define GET_LAYER_HPP
#include"layer.h"
#include "test_protobuf.pb.h"
#include "data_layer.h"

#include "innerproduct_layer.hpp"
#include "relu_layer.hpp"
#include "convolution_layer.hpp"
#include "pooling_layer.hpp"

namespace Test_cnn{
	shared_ptr<Test_layer> get_layer(const LayerParameter &_param){
		if (_param.type() == "Data"){
						
			return shared_ptr<Test_layer>(new Jpg_data_layer(_param));
		}
		else if (_param.type() == "Convolution"){
			return shared_ptr<Test_layer>(new Convolution_layer(_param));

		}
		else if (_param.type() == "ReLU"){
			return shared_ptr<Test_layer>(new Relu_layer(_param));

		}
		else if (_param.type() == "Pooling"){
			return shared_ptr<Test_layer>(new Pooling_layer(_param));

		}
		else if (_param.type() == "InnerProduct"){
			return shared_ptr<Test_layer>(new Innerproduct_layer(_param));

		}
		else {
			cout << "unkown layer type:" 
				<< _param.type() << endl;
			return NULL;
		}
	}
}


#endif