
SET CUDNN_VERSION=7
SET CUDNN_VERSION_DLL=7
SET CUDA_VERSION=9

rem SET CUDNN_VERSION=7
rem SET CUDNN_VERSION_DLL=7
rem SET CUDA_VERSION=8

SET CUDNN_INSTALL_PATH=C:\Users\pushkar\Downloads\cudnn-%CUDA_VERSION%.0-windows10-x64-v%CUDNN_VERSION%
SET CUDA_INSTALL_PATH=C:\"Program Files"\"NVIDIA GPU Computing Toolkit"

copy %CUDNN_INSTALL_PATH%\cuda\bin\cudnn64_%CUDNN_VERSION_DLL%.dll %CUDA_INSTALL_PATH%\CUDA\v%CUDA_VERSION%.0\bin
copy %CUDNN_INSTALL_PATH%\cuda\include\cudnn.h %CUDA_INSTALL_PATH%\CUDA\v%CUDA_VERSION%.0\include
copy %CUDNN_INSTALL_PATH%\cuda\lib\x64\cudnn.lib %CUDA_INSTALL_PATH%\CUDA\v%CUDA_VERSION%.0\lib\x64


