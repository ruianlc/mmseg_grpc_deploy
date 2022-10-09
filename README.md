# mmseg_grpc_deploy 
## 1.采用mmseg深度学习图像分割框架完成模型训练
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 是mmlab开发的基于pytorch构建的深度学习图像分割模型开发框架，能够让开发者以搭积木的形式，构建一整套从数据集准备、网络搭建、图像分割模型训练与调用的全流程方案。

## 2.模型调用GRPC服务搭建
采用grpc微服务框架搭建深度学习模型推理服务，主要包括：
* 消息传输格式定义：编写xxx.proto文件
* 通过grpc-tool对xxx.proto文件进行编译
* 定义server.py和client.py

### 2.1.定义消息传输格式
消息传输格式定义：编写xxx.proto文件。依据protobuf协议，根据消息数据类型，定义请求参数、返回参数，以及服务接口。

### 2.2.codegen编译生成grpc服务类
在codegen.py目录下，执行`python codegen.py`，对xxx.proto文件进行编译，生成可供grpc服务调用的request和response类。其中，在codegen.py文件中定义了多个传参，包括：
* '-I=./protos',          # 指定proto所在目录
* '--python_out=.',       # 生成服务py文件所在目录
* '--grpc_python_out=.',  # 生成服务grpc py文件所在目录
* './protos/dl_b64.proto',    # 指定proto文件

执行完成，生成dl_pb2.py和dl_pb2_grpc.py，分别表示：
* _pb2.py: 包含生成的请求和响应类
* _pb2_grpc.py: 包含生成的客户端和服务端类

### 2.3.编写server端代码
服务端定义了服务策略，包括地址、端口，并提供模型调用接口，以及请求和返回的输入输出定义。

### 2.4.编写client端调用代码
客户端定义了访问服务端的地址、端口，并传入请求参数，获得输出结果。

## 3.生产环境一键部署
该套服务在实际部署于windows平台，为了能够将整套服务快捷地部署于生产环境中。编写了Windows批处理脚本`easy_deploy.bat`来实现服务的一键部署。该脚本主要完成以下三件事情：
* 生产环境的配置
* 项目依赖包的下载
* 服务的开启

## 4.附注
### 4.1.依赖包
* grpcio==1.49.0
* grpcio-tools==1.48.1
* protobuf==3.19.4

### 4.2.gRPC
gRPC是Google开发的高性能、开源和通用的远程过程调用（Remote Procedure Calls, RP）系统，该系统基于ProtoBuf(Protocol Buffers) 序列化协议开发，且支持众多开发语言。面向服务端和移动端，基于 HTTP/2 协议传输，其大致请求流程为：
1、客户端（gRPC Stub）调用 A 方法，发起 RPC 调用。
2、对请求信息使用 Protobuf 进行对象序列化压缩（IDL）。
3、服务端（gRPC Server）接收到请求后，解码请求体，进行业务逻辑处理并返回。
4、对响应结果使用 Protobuf 进行对象序列化压缩（IDL）。
5、客户端接受到服务端响应，解码请求体。回调被调用的 A 方法，唤醒正在等待响应（阻塞）的客户端调用并返回响应结果。
