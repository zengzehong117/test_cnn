#ifndef   NET_CPP
#define   NET_CPP

#include "net.h"
#include "get_layer.hpp"


namespace Test_cnn{
	void Test_net::init(const NetParameter &_param){
		NetParameter param = _param;

		name_ = _param.name();		
		cout << "init net:"
			<< name_ << endl;
		layers_name_.resize(_param.layers_size());
		layers_parameter_.resize(_param.layers_size()); 
		//layers_.resize(_param.layers_size());
		top_blobs_.resize(_param.layers_size());
		bottom_blobs_.resize(_param.layers_size());

		for (int i = 0; i < _param.layers_size(); i++){
			layers_name_[i] = _param.layers(i).name();
			layers_parameter_[i] = param.layers(i);
			for (int j = 0; j < param.layers(i).top_size(); j++){
				if (map_blob_.find(_param.layers(i).top(j)) == map_blob_.end()){
					cout << "topname:" << _param.layers(i).top(j) << endl;
					map_blob_[_param.layers(i).top(j)] = shared_ptr<Blob>(new Blob());
				}
			}
			//map_blob_.
			// top_blobs_[i].resize(param.layers(i).top_size());

			for (int j = 0; j < param.layers(i).top_size(); j++){
				top_blobs_[i].push_back(map_blob_[param.layers(i).top(j)]);
			}
			
			for (int j = 0; j < param.layers(i).bottom_size(); j++){
				bottom_blobs_[i].push_back(map_blob_[param.layers(i).bottom(j)]);
			}

			layers_.push_back(get_layer(layers_parameter_[i]));
			
		}
		for (int i = 0; i < param.layers_size(); i++){
			layers_[i]->init_blobs(bottom_blobs_[i], top_blobs_[i]);

		}
		//layers_[0]->forward(bottom_blobs_[0], top_blobs_[0]);
		//layers_[1]->forward(bottom_blobs_[1], top_blobs_[1]);

		/*
		layers_[0]->init_blobs(bottom_blobs_[0], top_blobs_[0]);
		layers_[1]->init_blobs(bottom_blobs_[1], top_blobs_[1]);
		layers_[2]->init_blobs(bottom_blobs_[2], top_blobs_[2]);
		*/
		//layers_[0]->init_blobs(test_blob, test_blob);
		cout << "init net:"
			<< name_ << "success" << endl;
		return;
	}
	void Test_net::init_from_model(const Myblobs &_param){
		layers_[1]->init_multiplier(_param.blob(0));
		layers_[4]->init_multiplier(_param.blob(1));
		layers_[7]->init_multiplier(_param.blob(2));
		layers_[10]->init_multiplier(_param.blob(3));
		layers_[10]->init_multiplier(_param.blob(4));
		layers_[12]->init_multiplier(_param.blob(5));
		layers_[12]->init_multiplier(_param.blob(6));
	}
	void Test_net::forward(){
		for (int i = 0; i < 13; i++){
			layers_[i]->forward(bottom_blobs_[i], top_blobs_[i]);
		}
		/*
		layers_[0]->forward(bottom_blobs_[0], top_blobs_[0]);
		layers_[1]->forward(bottom_blobs_[1], top_blobs_[1]);
		layers_[2]->forward(bottom_blobs_[2], top_blobs_[2]);
		layers_[3]->forward(bottom_blobs_[3], top_blobs_[3]);
		*/
	}
	void Test_net::get_result(){
		float *a;
		a = top_blobs_[12][0]->get_data();
		cout << endl<<"The result is:"<<endl;
		for (int i = 0; i < 10; i++)
		{
			cout << a[i] << " " << endl;
		}
	}


}

#endif