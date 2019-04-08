#ifndef BLOB_CPP
#define BLOB_CPP

 // !BLOB_CPP


#include "blob.h"

namespace Test_cnn{
	Blob::Blob(const vector<int>&_shape){
		shape_ = _shape;
		lenth_ = 1;
		for (int i = 0; i < shape_.size(); i++){
			lenth_ *= shape_[i];
		}
		cout << "lenth_" << lenth_ << endl;
	}
	void Blob::set_shape(const vector<int> &_shape){
		shape_ = _shape;
		for (int i = 0; i < shape_.size(); i++){
			lenth_ *= shape_[i];
		}

		//data_.reset(new float(lenth_));
		//data_ = new float[lenth_];
		//data_.reset(new vector<float>(lenth_));
	}
	void Blob::push_one_shape(const int &i){
		shape_.push_back(i);
		lenth_ *= i;
	}
	float* Blob::creat_get_data_ptr(){
		assert(data_ == NULL);
		data_ = new float[lenth_];
		//data_ = (float*)malloc(lenth_*sizeof(float));
		assert(data_ != NULL);
		return data_;
	}
	float* Blob::creat_get_cudata_ptr(){
		assert(cu_data_ == NULL);
		cu_data_ = get_cuda_ptr(lenth_);
		//data_ = (float*)malloc(lenth_*sizeof(float));
		assert(cu_data_ != NULL);
		return cu_data_;
	}
	float* Blob::get_data(){
		assert(data_ != NULL);
		return data_;
	}
	float* Blob::is_null(){
		
		return data_;
	}

	
}


#endif // !BLOB_CPP
