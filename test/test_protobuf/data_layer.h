#ifndef DATA_LAYER_H
#define DATA_LAYER_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
#include "layer.h"


using namespace std;
namespace Test_cnn{
	class Jpg_data_layer : public Test_layer
	{
	public:
		Jpg_data_layer(){
			cout << "creat jpg_data layer." << endl;
		}
		Jpg_data_layer(const LayerParameter &_param){
			param_ = &_param;
			cout << "init jpg_data layer param." << endl;
		}
		//void test(int i);
		//void init_blobs(vector<Blob*>bottom_blobs, vector<Blob*>top_blobs);
		void init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);

		void forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs);


		~Jpg_data_layer(){}

	private:
		

	};


}


#endif