#ifndef DATA_LAYER_CPP
#define DATA_LAYER_CPP

#include "data_layer.h"
#include "global.hpp"


#include<opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>

namespace Test_cnn{
	//void Jpg_data_layer::ainit_blobs(int bottom_blobs, int top_blobs){	}
	
	void Jpg_data_layer::init_blobs(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){

		int i = param_->input_param().shape().dim(0);
		File_input_data_param param = param_->input_param();
		cout << param_->input_param().shape().dim(0) << endl;
		cout << param_->input_param().shape().dim(1) << endl;
		cout << param_->input_param().shape().dim(2) << endl;
		cout << param_->input_param().shape().dim(3) << endl;
		cout << i << endl;
		vector<int> shape;
		for (int i = 0; i < param_->input_param().shape().dim_size(); i++){
			shape.push_back(param_->input_param().shape().dim(i));
		}
		top_blobs[0]->set_shape(shape);
	//	float*top_data = top_blobs[0]->creat_get_data_ptr();

	}
	

	void Jpg_data_layer::forward(vector<shared_ptr<Blob>>&bottom_blobs, vector<shared_ptr<Blob>>&top_blobs){

		extern bool read_myblob(const string &_fpath, google::protobuf::Message *_net_param);

		/*
		BlobProto mean_data;
		string mean_path = "E:/mean.binaryproto";
		bool success=read_myblob(mean_path, &mean_data);
		*/
		
		char *img_path = "E:/cc.bmp";
		IplImage *my_jpg = cvLoadImage(img_path,1);

		//cvSaveImage("E:/cc.bmp", my_jpg);

		//IplImage *my_bmp = cvLoadImage("E:/cc.bmp", 1);
		int size = my_jpg->imageSize;

		//cvNamedWindow("example");
		//cvShowImage("example", my_jpg);

		float *top_data = top_blobs[0]->creat_get_data_ptr();

		float mean[] = { 125.3069178, 122.95039426, 113.86538316 };
//		cout << mean[0] << mean[1] << mean[2];

		//float *img_data_1 = new float[size];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 32 * 32; j++)
			{
				top_data[j + 1024 * i] =
					static_cast<float>(static_cast<unsigned char>(my_jpg->imageData[j * 3 + i]))
					- mean[i];
			}
		}

		/*

		for (int i = 0; i < size; i++)
		{
			top_data[i] = static_cast<float>(static_cast<unsigned char>
				(my_jpg->imageData[i])) - mean_data.data(i);
		}
		
		*/


	}

}







#endif // !DATA_LAYER_CPP
