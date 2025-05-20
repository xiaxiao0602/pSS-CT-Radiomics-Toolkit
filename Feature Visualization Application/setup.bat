@echo off
echo Setting up the environment...

REM 创建env目录（如果不存在）
if not exist env mkdir env
cd env

REM 下载Python嵌入式版本
echo Downloading Python...
powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-embed-amd64.zip' -OutFile 'python.zip'"

REM 解压Python
echo Extracting Python...
powershell -Command "Expand-Archive -Path 'python.zip' -DestinationPath 'python-3.9.13-embed-amd64' -Force"

REM 修改python39._pth文件以启用导入系统
echo Modifying python39._pth...
powershell -Command "(Get-Content python-3.9.13-embed-amd64\python39._pth) -replace '#import site', 'import site' | Set-Content python-3.9.13-embed-amd64\python39._pth"

REM 下载get-pip.py
echo Downloading pip...
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"

REM 创建目录结构
if not exist python-3.9.13-embed-amd64\Lib mkdir python-3.9.13-embed-amd64\Lib
if not exist python-3.9.13-embed-amd64\Lib\site-packages mkdir python-3.9.13-embed-amd64\Lib\site-packages

REM 设置环境变量
set PYTHON_HOME=%CD%\python-3.9.13-embed-amd64
set PATH=%PYTHON_HOME%;%PYTHON_HOME%\Scripts;%PATH%
set PYTHONPATH=%PYTHON_HOME%\Lib\site-packages

REM 安装pip
echo Installing pip...
%PYTHON_HOME%\python.exe get-pip.py --no-warn-script-location

REM 安装依赖
echo Installing dependencies...
%PYTHON_HOME%\Scripts\pip.exe install -r ..\requirements.txt --no-warn-script-location

REM 清理临时文件
del python.zip
del get-pip.py

cd ..
echo Setup completed successfully!
pause 