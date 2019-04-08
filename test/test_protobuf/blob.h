#ifndef  BLOB_H
#define  BLOB_H

#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "test_protobuf.pb.h"
//#include <boost/smart_ptr.hpp>
extern "C"  float *get_cuda_ptr(int _lenth);
using namespace std;

namespace Test_cnn{

	class Blob
	{
	public:
		Blob(){}
		~Blob(){
			assert(data_ != NULL);
			delete []data_;
		}

		Blob(const vector<int> &);
		void set_shape(const vector<int> &);
		void push_one_shape(const int &i);
		float* creat_get_data_ptr();
		float* creat_get_cudata_ptr();
		float * get_data();

		float* is_null();
		//const vector<int> get_shape(){			return shape_;		}
		int get_shape_0(){
			return shape_[0];
		}
		int get_shape_1(){
			return shape_[1];
		}
		int get_shape_2(){
			return shape_[2];
		}
		int get_shape_3(){
			return shape_[3];
		}
		int ge_lenth(){
			return lenth_;
		}
		
		void set_blob_from_model(const Myblob &_param){
			float *data = creat_get_data_ptr();
			for (int i = 0; i < lenth_; i++)
			{
				data[i] = _param.data(i);
			}
		}

	private:
		vector<int> shape_;
		int lenth_=1;
		float *data_;
		float *cu_data_;

		//boost::shared_array <float> data_;
		//shared_array<float> data_;
		//shared_ptr<vector<float>> data_;
	};
}

#endif