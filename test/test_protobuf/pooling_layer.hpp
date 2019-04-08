#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
#include "layer.h"


using namespace std;
namespace Test_cnn{
	class Pooling_layer : public Test_layer
	{
	public:
		Pooling_layer(){
			cout << "creat pooling layer." << endl;
		}
		Pooling_layer(const LayerParameter &_param){
			param_ = &_param;
			//multiplier_.reset(new Blob());
			cout << "creat pooling layer and init param." << endl;
		}
		~Pooling_layer(){}

		void init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);
		void forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);


	private:

	};


}


#endif