cd ..
# Download zip dataset from Google Drive
filename='libtorch-android.zip'
fileid='1wnlfDXEKqfOcOPqt--Zxda63-QonH9KB'
curl "https://drive.usercontent.google.com/download?id=1wnlfDXEKqfOcOPqt--Zxda63-QonH9KB&confirm=xxx" -o "libtorch-android.zip"

unzip ${filename}
rm ${filename}

echo "Done"
