cd ..
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
mv libtorch libtorch-linux
ln ../ios/Classes/native_pytorch.cpp ../linux/native_pytorch.cpp
echo "Done"
