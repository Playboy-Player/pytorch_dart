#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/script.h>
#include <unistd.h>
#include <string>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cstring>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define IS_WIN32
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

#ifdef IS_WIN32
#include <windows.h>
#endif

#if defined(__GNUC__)
// Attributes to prevent 'unused' function from being removed and to make it visible
#define FUNCTION_ATTRIBUTE __attribute__((visibility("default"))) __attribute__((used))
#elif defined(_MSC_VER)
// Marking a function for export
#define FUNCTION_ATTRIBUTE __declspec(dllexport)
#endif

long long int get_now()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void platform_log(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
#ifdef __ANDROID__
    __android_log_vprint(ANDROID_LOG_VERBOSE, "ndk", fmt, args);
#elif defined(IS_WIN32)
    char *buf = new char[4096];
    std::fill_n(buf, 4096, '\0');
    _vsprintf_p(buf, 4096, fmt, args);
    OutputDebugStringA(buf);
    delete[] buf;
#else
    vfprintf(stderr, fmt, args);
#endif
    va_end(args);
}

torch::jit::Module model;
std::stringstream printing_buffer;
struct TensorData
{
    int tensorBytesLen;
    int shapeBytesLen;
};
// 动态分配内存
// Note that you need to keep these alive while you're processing the lists on the Dart side
char *temp_print_buffer_ptr;
float *temp_out_data_ptr;

extern "C"
{

    FUNCTION_ATTRIBUTE
    void load_ml_model(const char *model_path)
    {
        model = torch::jit::load(model_path);
    }

    FUNCTION_ATTRIBUTE
    char *get_printing_buffer_and_clear()
    {
        const int length = printing_buffer.str().length();

        delete[] temp_print_buffer_ptr;
        temp_print_buffer_ptr = new char[length + 1];

        strcpy(temp_print_buffer_ptr, printing_buffer.str().c_str());

        printing_buffer.str("");

        return temp_print_buffer_ptr;
    }

    FUNCTION_ATTRIBUTE
    float **model_inference(float *input_data_ptr)
    {
        std::vector<torch::jit::IValue> inputs;

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto in_tensor = torch::from_blob(input_data_ptr, {17}, options);
        inputs.push_back(in_tensor);

        auto forward_output = model.forward(inputs);

        printing_buffer << "forward_output.tagKind(): " << forward_output.tagKind() << std::endl;

        // Note that here I'm only allowing the output to be tensor, but this can be easily changed
        // Also for simplicity I only assume that output tensor has only one dimension
        auto out_tensor = forward_output.toTensor();
        int tensor_lenght = out_tensor.sizes()[0];

        printing_buffer << "out_tensor.sizes(): " << out_tensor.sizes() << std::endl;

        delete[] temp_out_data_ptr;
        temp_out_data_ptr = new float[tensor_lenght];
        std::memcpy(temp_out_data_ptr, out_tensor.data_ptr<float>(), sizeof(float) * tensor_lenght);

        // Maybe not so pretty, but you need to somehow return the array with information about number of elements
        float **out_data_with_length = new float *[2];
        float *out_data_length = new float[1];
        out_data_length[0] = tensor_lenght;
        out_data_with_length[0] = temp_out_data_ptr;
        out_data_with_length[1] = out_data_length;

        return out_data_with_length;
    }

    FUNCTION_ATTRIBUTE
    struct TensorData transpose_int(int tensorBytesCount, int *rawTensorBytes, int shapeBytesCount, int *rawShapeBytes, int **rawOutputTensor, int **rawOutputShape, int dim1, int dim2)
    {

        std::vector<int> tensorBuffer(rawTensorBytes, rawTensorBytes + tensorBytesCount);
        std::vector<int> shapeBuffer(rawShapeBytes, rawShapeBytes + shapeBytesCount);
        std::vector<long> longShapeVector(shapeBuffer.begin(), shapeBuffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt);
        torch::IntArrayRef s(longShapeVector.data(), longShapeVector.size()); // 设置返回的tensor的大小
        torch::Tensor t = torch::from_blob(tensorBuffer.data(), s, opts).clone();

        auto tensor = t.transpose(dim1, dim2);

        // 获取连续的张量
        torch::Tensor contiguous_tensor = tensor.contiguous();

        // 转换为std::vector
        std::vector<int> tensorbuf(contiguous_tensor.data_ptr<int>(), contiguous_tensor.data_ptr<int>() + contiguous_tensor.numel());

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (int *)malloc(tensorbuf.size() * sizeof(int));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(int));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();
        std::cerr << tensorData.tensorBytesLen << std::endl
                  << tensorData.shapeBytesLen << std::endl;
        return tensorData;
    }
    FUNCTION_ATTRIBUTE
    struct TensorData transpose_float(int tensorBytesCount, double *rawTensorBytes, int shapeBytesCount, int *rawShapeBytes, double **rawOutputTensor, int **rawOutputShape, int dim1, int dim2)
    {

        std::vector<double> tensorBuffer(rawTensorBytes, rawTensorBytes + tensorBytesCount);
        std::vector<int> shapeBuffer(rawShapeBytes, rawShapeBytes + shapeBytesCount);
        std::vector<long> longShapeVector(shapeBuffer.begin(), shapeBuffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
        torch::IntArrayRef s(longShapeVector.data(), longShapeVector.size()); // 设置返回的tensor的大小
        torch::Tensor t = torch::from_blob(tensorBuffer.data(), s, opts).clone();
        for (int i = 0; i < tensorBuffer.size(); i++)
        {
            std::cerr << tensorBuffer[i] << std::endl;
        }

        auto tensor = t.transpose(dim1, dim2);

        // 获取连续的张量
        torch::Tensor contiguous_tensor = tensor.contiguous();

        // 转换为std::vector
        std::vector<double> tensorbuf(contiguous_tensor.data_ptr<double>(), contiguous_tensor.data_ptr<double>() + contiguous_tensor.numel());
        for (int i = 0; i < tensorbuf.size(); i++)
        {
            std::cerr << tensorbuf[i] << std::endl;
        }
        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (double *)malloc(tensorbuf.size() * sizeof(double));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(double));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }

    FUNCTION_ATTRIBUTE
    struct TensorData reshape_int(int tensorBytesCount, int *rawTensorBytes, int shapeBytesCount, int *rawShapeBytes, int targetShapeBytesCount, int *rawTargetShapeBytes, int **rawOutputTensor, int **rawOutputShape, int dim1, int dim2)
    {

        std::vector<int> tensorBuffer(rawTensorBytes, rawTensorBytes + tensorBytesCount);
        std::vector<int> shapeBuffer(rawTargetShapeBytes, rawTargetShapeBytes + targetShapeBytesCount);
        std::vector<long> longShapeVector(shapeBuffer.begin(), shapeBuffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt);
        torch::IntArrayRef s(longShapeVector.data(), longShapeVector.size()); // 设置返回的tensor的大小
        torch::Tensor tensor = torch::from_blob(tensorBuffer.data(), s, opts).clone();

        std::vector<int> tensorbuf(tensor.data_ptr<int>(), tensor.data_ptr<int>() + tensor.numel()); // 将tensor转换为vector

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (int *)malloc(tensorbuf.size() * sizeof(int));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(int));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }

    FUNCTION_ATTRIBUTE
    struct TensorData reshape_float(int tensorBytesCount, double *rawTensorBytes, int shapeBytesCount, int *rawShapeBytes, int targetShapeBytesCount, int *rawTargetShapeBytes, double **rawOutputTensor, int **rawOutputShape, int dim1, int dim2)
    {

        std::vector<double> tensorBuffer(rawTensorBytes, rawTensorBytes + tensorBytesCount);
        std::vector<int> shapeBuffer(rawTargetShapeBytes, rawTargetShapeBytes + targetShapeBytesCount);
        std::vector<long> longShapeVector(shapeBuffer.begin(), shapeBuffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
        torch::IntArrayRef s(longShapeVector.data(), longShapeVector.size()); // 设置返回的tensor的大小
        torch::Tensor tensor = torch::from_blob(tensorBuffer.data(), s, opts).clone();

        std::vector<double> tensorbuf(tensor.data_ptr<double>(), tensor.data_ptr<double>() + tensor.numel()); // 将tensor转换为vector

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (double *)malloc(tensorbuf.size() * sizeof(double));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(double));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }

    struct TensorData subtract_int(int tensor1BytesCount, int *rawTensor1Bytes, int shape1BytesCount, int *rawShape1Bytes, int tensor2BytesCount, int *rawTensor2Bytes, int shape2BytesCount, int *rawShape2Bytes, int **rawOutputTensor, int **rawOutputShape, int alpha)
    {

        std::vector<int> tensor1Buffer(rawTensor1Bytes, rawTensor1Bytes + tensor1BytesCount);
        std::vector<int> shape1Buffer(rawShape1Bytes, rawShape1Bytes + shape1BytesCount);
        std::vector<long> longShape1Vector(shape1Buffer.begin(), shape1Buffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt);
        torch::IntArrayRef s1(longShape1Vector.data(), longShape1Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t1 = torch::from_blob(tensor1Buffer.data(), s1, opts).clone();

        std::vector<int> tensor2Buffer(rawTensor2Bytes, rawTensor2Bytes + tensor2BytesCount);
        std::vector<int> shape2Buffer(rawShape2Bytes, rawShape2Bytes + shape2BytesCount);
        std::vector<long> longShape2Vector(shape2Buffer.begin(), shape2Buffer.end());

        torch::IntArrayRef s2(longShape2Vector.data(), longShape2Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t2 = torch::from_blob(tensor2Buffer.data(), s2, opts).clone();

        auto tensor = torch::sub(t1, t2, alpha);

        std::vector<int> tensorbuf(tensor.data_ptr<int>(), tensor.data_ptr<int>() + tensor.numel()); // 将tensor转换为vector

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (int *)malloc(tensorbuf.size() * sizeof(int));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(int));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }
    FUNCTION_ATTRIBUTE
    struct TensorData sub_float(int tensor1BytesCount, double *rawTensor1Bytes, int shape1BytesCount, int *rawShape1Bytes, int tensor2BytesCount, double *rawTensor2Bytes, int shape2BytesCount, int *rawShape2Bytes, double **rawOutputTensor, int **rawOutputShape, int alpha)
    {

        std::vector<double> tensor1Buffer(rawTensor1Bytes, rawTensor1Bytes + tensor1BytesCount);
        std::vector<int> shape1Buffer(rawShape1Bytes, rawShape1Bytes + shape1BytesCount);
        std::vector<long> longShape1Vector(shape1Buffer.begin(), shape1Buffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
        torch::IntArrayRef s1(longShape1Vector.data(), longShape1Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t1 = torch::from_blob(tensor1Buffer.data(), s1, opts).clone();

        std::vector<double> tensor2Buffer(rawTensor2Bytes, rawTensor2Bytes + tensor2BytesCount);
        std::vector<int> shape2Buffer(rawShape2Bytes, rawShape2Bytes + shape2BytesCount);
        std::vector<long> longShape2Vector(shape2Buffer.begin(), shape2Buffer.end());

        torch::IntArrayRef s2(longShape2Vector.data(), longShape2Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t2 = torch::from_blob(tensor2Buffer.data(), s2, opts).clone();
        auto tensor = torch::sub(t1, t2, alpha);

        std::vector<double> tensorbuf(tensor.data_ptr<double>(), tensor.data_ptr<double>() + tensor.numel()); // 将tensor转换为vector

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (double *)malloc(tensorbuf.size() * sizeof(double));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(double));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }

    struct TensorData add_int(int tensor1BytesCount, int *rawTensor1Bytes, int shape1BytesCount, int *rawShape1Bytes, int tensor2BytesCount, int *rawTensor2Bytes, int shape2BytesCount, int *rawShape2Bytes, int **rawOutputTensor, int **rawOutputShape, int alpha)
    {

        std::vector<int> tensor1Buffer(rawTensor1Bytes, rawTensor1Bytes + tensor1BytesCount);
        std::vector<int> shape1Buffer(rawShape1Bytes, rawShape1Bytes + shape1BytesCount);
        std::vector<long> longShape1Vector(shape1Buffer.begin(), shape1Buffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kInt);
        torch::IntArrayRef s1(longShape1Vector.data(), longShape1Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t1 = torch::from_blob(tensor1Buffer.data(), s1, opts).clone();

        std::vector<int> tensor2Buffer(rawTensor2Bytes, rawTensor2Bytes + tensor2BytesCount);
        std::vector<int> shape2Buffer(rawShape2Bytes, rawShape2Bytes + shape2BytesCount);
        std::vector<long> longShape2Vector(shape2Buffer.begin(), shape2Buffer.end());

        torch::IntArrayRef s2(longShape2Vector.data(), longShape2Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t2 = torch::from_blob(tensor2Buffer.data(), s2, opts).clone();

        auto tensor = torch::add(t1, t2, alpha);

        std::vector<int> tensorbuf(tensor.data_ptr<int>(), tensor.data_ptr<int>() + tensor.numel()); // 将tensor转换为vector

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (int *)malloc(tensorbuf.size() * sizeof(int));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(int));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }
    FUNCTION_ATTRIBUTE
    struct TensorData add_float(int tensor1BytesCount, double *rawTensor1Bytes, int shape1BytesCount, int *rawShape1Bytes, int tensor2BytesCount, double *rawTensor2Bytes, int shape2BytesCount, int *rawShape2Bytes, double **rawOutputTensor, int **rawOutputShape, int alpha)
    {

        std::vector<double> tensor1Buffer(rawTensor1Bytes, rawTensor1Bytes + tensor1BytesCount);
        std::vector<int> shape1Buffer(rawShape1Bytes, rawShape1Bytes + shape1BytesCount);
        std::vector<long> longShape1Vector(shape1Buffer.begin(), shape1Buffer.end());
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
        torch::IntArrayRef s1(longShape1Vector.data(), longShape1Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t1 = torch::from_blob(tensor1Buffer.data(), s1, opts).clone();

        std::vector<double> tensor2Buffer(rawTensor2Bytes, rawTensor2Bytes + tensor2BytesCount);
        std::vector<int> shape2Buffer(rawShape2Bytes, rawShape2Bytes + shape2BytesCount);
        std::vector<long> longShape2Vector(shape2Buffer.begin(), shape2Buffer.end());

        torch::IntArrayRef s2(longShape2Vector.data(), longShape2Vector.size()); // 设置返回的tensor的大小
        torch::Tensor t2 = torch::from_blob(tensor2Buffer.data(), s2, opts).clone();
        auto tensor = torch::add(t1, t2, alpha);

        std::vector<double> tensorbuf(tensor.data_ptr<double>(), tensor.data_ptr<double>() + tensor.numel()); // 将tensor转换为vector

        torch::Tensor rawoutputshape = torch::_shape_as_tensor(tensor);
        torch::Tensor outputshape = rawoutputshape.to(torch::kInt);

        std::vector<int> shapebuf(outputshape.data_ptr<int>(), outputshape.data_ptr<int>() + outputshape.numel());

        *rawOutputTensor = (double *)malloc(tensorbuf.size() * sizeof(double));
        memcpy(*rawOutputTensor, tensorbuf.data(), tensorbuf.size() * sizeof(double));

        *rawOutputShape = (int *)malloc(shapebuf.size() * sizeof(int));
        memcpy(*rawOutputShape, shapebuf.data(), shapebuf.size() * sizeof(int));
        struct TensorData tensorData;

        tensorData.tensorBytesLen = tensorbuf.size();
        tensorData.shapeBytesLen = shapebuf.size();

        return tensorData;
    }
}