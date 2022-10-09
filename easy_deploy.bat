::-- Script to automate deploy deep learning model inferiing service to new environmenrt.
@REM 一键部署项目执行环境。使用方法：
@REM  - 1.手动安装anaconda安装包，直接next下去，直到安装完成。（注意有一步需要将其添加到环境变量）
@REM  - 2.在项目目录下打开anaconda窗口，在命令行输入脚本执行命令：.\easy_deploy.bat（windows批处理脚本）
@REM 优点在于，即可以执行DOS命令，也可以执行python环境命令

echo ============开始部署============

:: 1、配置anaconda的python执行环境
:: 根据conda安装地址相应修改
set CONDAPATH=D:\ProgramData\Anaconda
call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%

:: 2、配置国内源，提高下载速度
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
call conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

:: 3、安装项目依赖包
pip install numpy terminaltables matplotlib tqdm pandas timm Pillow grpcio==1.49.0 protobuf==4.21.7 opencv_python_headless==4.5.5.64
pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
echo ============依赖包下载完成！============

:: 4、开启grpc服务端
python dl_server.py

echo ============服务启动完成！============