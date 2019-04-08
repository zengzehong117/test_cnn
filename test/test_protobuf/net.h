#ifndef   NET_H
#define   NET_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "blob.h"
#include "layer.h"
#include "data_layer.h"


//#include "E:\NugetPackages\gflags.2.1.2.1\build\native\include\gflags\gflags.h"
//#include "E:\NugetPackages\glog.0.3.3.0\build\native\include\glog/logging.h"

#include "test_protobuf.pb.h"
using namespace std;
namespace Test_cnn{
	class Test_net
	{
	public:

		Test_net(){

		}

		//Test_net(const NetParameter param){}
		void init(const NetParameter &param);
		void init_from_model(const Myblobs &_param);
		void setup();
		void forward();
		void get_result();

		
		~Test_net(){}

	private:


		string name_;
		vector<string>layers_name_;
		vector<LayerParameter>layers_parameter_;
		vector < shared_ptr<Test_layer>> layers_;
		vector<vector<shared_ptr<Blob>>> top_blobs_;
		vector<vector<shared_ptr<Blob>>> bottom_blobs_;
		map<string, shared_ptr<Blob>> map_blob_;
		shared_ptr<Blob> test;

	//	Blob test;
	//	Test_layer *layer;
	//	Test_layer *test=new Jpg_data_layer();
		vector<shared_ptr<Blob>> Intermediate_blob_;
		vector<vector<int>> Intermediate_blob_shape_;

	};
}

#endif