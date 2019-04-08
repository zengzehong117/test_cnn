#ifndef   LAYER_H     
#define   LAYER_H 

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
#include "blob.h"
/*#include "data_layer.h"
#include "innerproduct_layer.hpp"
#include "relu_layer.hpp"
#include "convolution_layer.hpp"
#include "pooling_layer.hpp"

*/
using namespace std;
namespace Test_cnn{
	class Test_layer
	{
	public:
		Test_layer(){
			cout << "creat father layer." << endl;
		}
		Test_layer(const LayerParameter *_param){
			param_ = _param;
			cout << "init layer param." << endl;

		}
		~Test_layer(){}
		void init(void);
		virtual void init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){}
		virtual void init_multiplier(const Myblob &_param){}
		virtual void forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){}


	protected:
		string name_;
		string type_;
		vector<string> bottom_names_;
		vector<string> top_names_;
		shared_ptr<Blob> multiplier_;
		shared_ptr<Blob> bias_multiplier_;

	//	Blob multiplier_;
		const LayerParameter *param_;
	};

}
/*
class TYPE_DATA
{
public:
	TYPE_DATA(){}
	~TYPE_DATA(){}

private:

};

class TYPE_CONV
{
public:
	TYPE_CONV(){}
	~TYPE_CONV(){}
	
private:

};

class TYPE:public TYPE_CONV{
public:
	TYPE(){}
	~TYPE(){}

private:

};
*/


#endif