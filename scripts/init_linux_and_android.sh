

wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
pwd
mv libtorch libtorch-linux
# Download zip dataset from Google Drive
filename='libtorch-android.zip'
fileid='1wnlfDXEKqfOcOPqt--Zxda63-QonH9KB'
curl "https://drive.usercontent.google.com/download?id=1wnlfDXEKqfOcOPqt--Zxda63-QonH9KB&confirm=xxx" -o "libtorch-android.zip"

unzip ${filename}
ln ./ios/Classes/native_pytorch.cpp ./linux/native_pytorch.cpp
echo "Done"
