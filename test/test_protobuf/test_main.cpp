#ifndef   MAIN_CPP      
#define   MAIN_CPP    

#include "global.hpp"

#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include<string>
#include "net.h"
#include <io.h>
//e:\NugetPackages\protobuf - v120.2.6.1\build\native\include\google\protobuf\io\zero_copy_stream_impl.h
#include <google/protobuf/stubs/common.h>

#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>
#include <assert.h>
#include "test_protobuf.pb.h"
//#include <gflags/gflags.h>
//#include <glog/logging.h>
//#include <gflags/gflags.h> // E:\NugetPackages\gflags.2.1.2.1\build\native\include\
 //E:\NugetPackages\glog.0.3.3.0\build\native\include\glog



#include <google/protobuf/io/zero_copy_stream.h>


//#include "convolution_layer.cu"



//#include <algorithm>
// NOLINT(readability/streams)
//#include <vector>


using namespace std;
using namespace Test_cnn;
using google::protobuf::io::FileInputStream;
extern "C" int cc();
//#define _CRT_SECURE_NO_DEPRECATE 1

//#define _CRT_NONSTDC_NO_DEPRECATE 1 


//E:\caffe-windows\examples\cifar10\20-res\solver.prototxt
bool read_param(const string &_fpath, NetParameter *_net_param){
	//char char_filepath[100]="E:\Projects\res20_cifar_train_test.prototxt";
	//	strcpy_s(char_filepath,_fpath.c_str());

	//string ccccc = "E:\\a.txt";//E:\caffe-windows\examples\cifar10\20-res\res20_cifar_train_test.prototxt";
	char char_filepath[100];
	strcpy_s(char_filepath, _fpath.c_str());

	int file_id = _open(char_filepath, O_RDONLY);
	assert(file_id > 2);
	//CHECK_NE(file_id, -1) << "Can't open netparamfile:" << _fpath;
	google::protobuf::io::FileInputStream *file_stream =
		new google::protobuf::io::FileInputStream(file_id);
	bool is_success = google::protobuf::TextFormat::Parse(file_stream, _net_param);
	delete file_stream;
	_close(file_id);
	return 1;

}

bool read_myblob(const string &_fpath, google::protobuf::Message *_net_param){
	//char char_filepath[100]="E:\Projects\res20_cifar_train_test.prototxt";
	//	strcpy_s(char_filepath,_fpath.c_str());

	/*string ccccc = "E:\\a.txt";//E:\caffe-windows\examples\cifar10\20-res\res20_cifar_train_test.prototxt";*/
	char char_filepath[100];
	strcpy_s(char_filepath, _fpath.c_str());

	int file_id = _open(char_filepath, O_RDONLY | O_BINARY);
	assert(file_id > 2);
	//CHECK_NE(file_id, -1) << "Can't open netparamfile:" << _fpath;

	google::protobuf::io::ZeroCopyInputStream *raw_input = new google::protobuf::io::FileInputStream(file_id);
	google::protobuf::io::CodedInputStream* coded_input = new google::protobuf::io::CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(2147483647, 536870912);
	bool success = _net_param->ParseFromCodedStream(coded_input);
	/*
	google::protobuf::io::FileInputStream *file_stream =
	new google::protobuf::io::FileInputStream(file_id);
	bool is_success = google::protobuf::TextFormat::Parse(file_stream, _net_param);
	*/
	delete coded_input;
	delete raw_input;
	_close(file_id);
	return success;

}

/*bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
#if defined (_MSC_VER)  // for MSC compiler binary flag needs to be specified
int fd = open(filename, O_RDONLY | O_BINARY);
#else
int fd = open(filename, O_RDONLY);
#endif
CHECK_NE(fd, -1) << "File not found: " << filename;
ZeroCopyInputStream* raw_input = new FileInputStream(fd);
CodedInputStream* coded_input = new CodedInputStream(raw_input);
coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

bool success = proto->ParseFromCodedStream(coded_input);

delete coded_input;
delete raw_input;
close(fd);
return success;
}


bool ReadProtoFromTextFile(const char* filename, Message* proto) {
int fd = open(filename, O_RDONLY);
CHECK_NE(fd, -1) << "File not found: " << filename;
FileInputStream* input = new FileInputStream(fd);
bool success = google::protobuf::TextFormat::Parse(input, proto);
delete input;
close(fd);
return success;
}
*/
//extern "C" int cc();

void main(int argc, char ** argv)
{

	cc();
	NetParameter net_param;
	
	std::cout << argv[1];
	read_param(argv[1], &net_param);
	//CHECK(read_param(argv[1], &net_param)) << "read from txt successful.";
	Test_net test_net;
	test_net.init(net_param);

	string mypath = "E:/cc.model";
	Myblobs multipliers;
	bool success=read_myblob(mypath, &multipliers);
	test_net.init_from_model(multipliers);

	test_net.forward();
	//cout << test_net.top_blobs_[12]
	test_net.get_result();
	system("pause");
	//Test_net test_net2=new Test_net(net_param);

	//test_net.set_name("net1");


	
}


#endif
