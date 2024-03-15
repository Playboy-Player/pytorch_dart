import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'dart:developer' as dev;
import 'package:pytorch_dart/cString.dart';

import 'native_pytorch_platform_interface.dart';

final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libnative_pytorch.so')
    : DynamicLibrary.process();

final void Function(Pointer<Uint8> modelPath) nativeLoadMlModel = nativeLib
    .lookup<NativeFunction<Void Function(Pointer<Uint8> modelPath)>>(
        "load_ml_model")
    .asFunction();

final Pointer<Pointer<Float>> Function(Pointer<Float> inputData)
    nativeModelInference = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Pointer<Float>> Function(
                    Pointer<Float> inputData)>>("model_inference")
        .asFunction();

final Pointer<Uint8> Function() nativeGetPrintingBufferAndClear = nativeLib
    .lookup<NativeFunction<Pointer<Uint8> Function()>>(
        "get_printing_buffer_and_clear")
    .asFunction();

final TensorData Function(
        int tensor1BytesCount,
        ffi.Pointer<ffi.Int32> tensor1Data,
        int shape1BytesCount,
        ffi.Pointer<ffi.Int32> shape1Data,
        int tensor2BytesCount,
        ffi.Pointer<ffi.Int32> tensor2Data,
        int shape2BytesCount,
        ffi.Pointer<ffi.Int32> shape2Data,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
        int alpha
        ) _add_Int =
    nativeLib
        .lookup<
            ffi.NativeFunction<
                TensorData Function(
                    ffi.Int32 tensor1BytesCount,
                    ffi.Pointer<ffi.Int32> tensor1Data,
                    ffi.Int32 shape1BytesCount,
                    ffi.Pointer<ffi.Int32> shape1Data,
                    ffi.Int32 tensor2BytesCount,
                    ffi.Pointer<ffi.Int32> tensor2Data,
                    ffi.Int32 shape2BytesCount,
                    ffi.Pointer<ffi.Int32> shape2Data,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
                    ffi.Int32 alpha
                    )>>('add_int')
        .asFunction();


final TensorData Function(
        int tensor1BytesCount,
        ffi.Pointer<ffi.Double> tensor1Data,
        int shape1BytesCount,
        ffi.Pointer<ffi.Int32> shape1Data,
        int tensor2BytesCount,
        ffi.Pointer<ffi.Double> tensor2Data,
        int shape2BytesCount,
        ffi.Pointer<ffi.Int32> shape2Data,
        ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
        int alpha
        ) _add_Float =
    nativeLib
        .lookup<
            ffi.NativeFunction<
                TensorData Function(
                    ffi.Int32 tensor1BytesCount,
                    ffi.Pointer<ffi.Double> tensor1Data,
                    ffi.Int32 shape1BytesCount,
                    ffi.Pointer<ffi.Int32> shape1Data,
                    ffi.Int32 tensor2BytesCount,
                    ffi.Pointer<ffi.Double> tensor2Data,
                    ffi.Int32 shape2BytesCount,
                    ffi.Pointer<ffi.Int32> shape2Data,
                    ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
                    ffi.Int32 alpha
                    )>>('add_float')
        .asFunction();


final TensorData Function(
        int tensor1BytesCount,
        ffi.Pointer<ffi.Int32> tensor1Data,
        int shape1BytesCount,
        ffi.Pointer<ffi.Int32> shape1Data,
        int tensor2BytesCount,
        ffi.Pointer<ffi.Int32> tensor2Data,
        int shape2BytesCount,
        ffi.Pointer<ffi.Int32> shape2Data,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
        int alpha
        ) _sub_Int =
    nativeLib
        .lookup<
            ffi.NativeFunction<
                TensorData Function(
                    ffi.Int32 tensor1BytesCount,
                    ffi.Pointer<ffi.Int32> tensor1Data,
                    ffi.Int32 shape1BytesCount,
                    ffi.Pointer<ffi.Int32> shape1Data,
                    ffi.Int32 tensor2BytesCount,
                    ffi.Pointer<ffi.Int32> tensor2Data,
                    ffi.Int32 shape2BytesCount,
                    ffi.Pointer<ffi.Int32> shape2Data,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
                    ffi.Int32 alpha
                    )>>('sub_int')
        .asFunction();


final TensorData Function(
        int tensor1BytesCount,
        ffi.Pointer<ffi.Double> tensor1Data,
        int shape1BytesCount,
        ffi.Pointer<ffi.Int32> shape1Data,
        int tensor2BytesCount,
        ffi.Pointer<ffi.Double> tensor2Data,
        int shape2BytesCount,
        ffi.Pointer<ffi.Int32> shape2Data,
        ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
        int alpha
        ) _sub_Float =
    nativeLib
        .lookup<
            ffi.NativeFunction<
                TensorData Function(
                    ffi.Int32 tensor1BytesCount,
                    ffi.Pointer<ffi.Double> tensor1Data,
                    ffi.Int32 shape1BytesCount,
                    ffi.Pointer<ffi.Int32> shape1Data,
                    ffi.Int32 tensor2BytesCount,
                    ffi.Pointer<ffi.Double> tensor2Data,
                    ffi.Int32 shape2BytesCount,
                    ffi.Pointer<ffi.Int32> shape2Data,
                    ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
                    ffi.Int32 alpha
                    )>>('sub_float')
        .asFunction();

final TensorData Function(
  int inBytesCount,
  ffi.Pointer<ffi.Int32> tensorData,
  int shapeBytesCount,
  ffi.Pointer<ffi.Int32> shapeData,
  int targetShapeBytesCount,
  ffi.Pointer<ffi.Int32> targetShapeData,
  ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
  ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
) _reshape_Int = nativeLib
    .lookup<
        ffi.NativeFunction<
            TensorData Function(
              ffi.Int32 inBytesCount,
              ffi.Pointer<ffi.Int32> tensorData,
              ffi.Int32 shapeBytesCount,
              ffi.Pointer<ffi.Int32> shapeData,
              ffi.Int32 targetShapeBytesCount,
              ffi.Pointer<ffi.Int32> targetShapeData,
              ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
              ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
            )>>('reshape_int')
    .asFunction();

final TensorData Function(
  int inBytesCount,
  ffi.Pointer<ffi.Double> tensorData,
  int shapeBytesCount,
  ffi.Pointer<ffi.Int32> shapeData,
  int targetShapeBytesCount,
  ffi.Pointer<ffi.Int32> targetShapeData,
  ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
  ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
) _reshape_Float = nativeLib
    .lookup<
        ffi.NativeFunction<
            TensorData Function(
              ffi.Int32 inBytesCount,
              ffi.Pointer<ffi.Double> tensorData,
              ffi.Int32 shapeBytesCount,
              ffi.Pointer<ffi.Int32> shapeData,
              ffi.Int32 targetShapeBytesCount,
              ffi.Pointer<ffi.Int32> targetShapeData,
              ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
              ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
            )>>('reshape_float')
    .asFunction();

final TensorData Function(
        int inBytesCount,
        ffi.Pointer<ffi.Int32> tensorData,
        int shapeBytesCount,
        ffi.Pointer<ffi.Int32> shapeData,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
        int dim1,
        int dim2) _transpose_Int =
    nativeLib
        .lookup<
            ffi.NativeFunction<
                TensorData Function(
                    ffi.Int32 inBytesCount,
                    ffi.Pointer<ffi.Int32> tensorData,
                    ffi.Int32 shapeBytesCount,
                    ffi.Pointer<ffi.Int32> shapeData,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensor,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
                    ffi.Int32 dim1,
                    ffi.Int32 dim2)>>('transpose_int')
        .asFunction();
final TensorData Function(
        int inBytesCount,
        ffi.Pointer<ffi.Double> tensorData,
        int shapeBytesCount,
        ffi.Pointer<ffi.Int32> shapeData,
        ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
        int dim1,
        int dim2) _transpose_Float =
    nativeLib
        .lookup<
            ffi.NativeFunction<
                TensorData Function(
                    ffi.Int32 inBytesCount,
                    ffi.Pointer<ffi.Double> tensorData,
                    ffi.Int32 shapeBytesCount,
                    ffi.Pointer<ffi.Int32> shapeData,
                    ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensor,
                    ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShape,
                    ffi.Int32 dim1,
                    ffi.Int32 dim2)>>('transpose_float')
        .asFunction();

class NativePytorch {
  Future<String?> getPlatformVersion() {
    return NativePytorchPlatform.instance.getPlatformVersion();
  }
}

final class TensorData extends ffi.Struct {
  @ffi.Int32()
  external int tensorbytesLen;

  @ffi.Int32()
  external int shapebytesLen;
}

class Tensor {
  List<num>? _tensorData;
  List<int>? _shape;
  Tensor(List<num> tensordata, List<int> shape) {
    // 构造函数接受 List<num> 类型的参数
    _tensorData = tensordata;

    int _shapeProduct =
        shape!.fold(1, (previousValue, element) => previousValue * element);

    if (_shapeProduct == _tensorData!.length) {
      _shape = shape;
    } else {
      throw Exception("wrong shape");
    }
  }

  List<num> get tensorData => _tensorData!;
  List<int> get shape => _shape!;

  Tensor transpose(int dim1, int dim2) {
   if (this._tensorData != null &&
      this._shape != null &&
      dim1 < this._shape!.length &&
      dim2 < this._shape!.length) {
    if (this._tensorData![1] is int) {
      DateTime startTime;
      startTime = DateTime.now();

      List<int> tensorData =
          this._tensorData!.map((num) => num.toInt()).toList();
      Int32List tensorList = Int32List.fromList(tensorData);
      List<int> shapeData = this._shape!.map((num) => num.toInt()).toList();
      Int32List shapeList = Int32List.fromList(shapeData);
      ffi.Pointer<ffi.Int32> tensorPtr =
          malloc.allocate(tensorList.lengthInBytes);
      ffi.Pointer<ffi.Int32> shapePtr =
          malloc.allocate(shapeList.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensorPtr.asTypedList(tensorList.lengthInBytes).setAll(0, tensorList);
      shapePtr.asTypedList(shapeList.lengthInBytes).setAll(0, shapeList);
      
      TensorData rawtensorData = _transpose_Int(
          this._tensorData!.length,
          tensorPtr,
          this._shape!.length,
          shapePtr,
          outputTensorLen,
          outputShapeLen,
          dim1,
          dim2);
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Int32> cppTensorPtr = outputTensorLen.value;
      Int32List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());

      dev.log('${rawtensorData.tensorbytesLen}');
      dev.log('${rawtensorData.shapebytesLen}');
      dev.log('${outputTensor._tensorData}');
      dev.log('${outputTensor._shape}');
      if (outputTensor != null) {
        return outputTensor;
      } else {
        throw Exception("output Tensor is empty");
      }
    } else if (this._tensorData![1] is double) {
      DateTime startTime;
      startTime = DateTime.now();

      List<double> tensorData =
          this._tensorData!.map((num) => num.toDouble()).toList();
      Float64List tensorList = Float64List.fromList(tensorData);
      List<int> shapeData = this._shape!.map((num) => num.toInt()).toList();
      Int32List shapeList = Int32List.fromList(shapeData);
      ffi.Pointer<ffi.Double> tensorPtr =
          malloc.allocate(tensorList.lengthInBytes);
      ffi.Pointer<ffi.Int32> shapePtr =
          malloc.allocate(shapeList.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensorPtr.asTypedList(tensorList.lengthInBytes).setAll(0, tensorList);
      shapePtr.asTypedList(shapeList.lengthInBytes).setAll(0, shapeList);
   
      TensorData rawtensorData = _transpose_Float(
          this._tensorData!.length,
          tensorPtr,
          this._shape!.length,
          shapePtr,
          outputTensorLen,
          outputShapeLen,
          dim1,
          dim2);
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Double> cppTensorPtr = outputTensorLen.value;
      Float64List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());
      DateTime endTime = DateTime.now();
      Duration elapsedTime = endTime.difference(startTime);
      dev.log(
          'Function transpose took ${elapsedTime.inMilliseconds} milliseconds to execute.');

      return outputTensor;
    }
    // 如果Tensor数据不是预期类型
    else {
      throw Exception("Tensor数据不是预期的类型");
    }
  } else {
    // 如果Tensor数据或形状为空
    throw Exception("input Tensor is empty");
  }
  }

  Tensor reshape(List<int> targetShape) {
    //shapeData,shapeList and shapePtr are useless here because I don't actually use torch::reshape to reshape
    int targetShapeProduct = targetShape!
        .fold(1, (previousValue, element) => previousValue * element);
    if (this._tensorData != null &&
        this._shape != null &&
        targetShapeProduct == this.tensorData.length) {
      if (this._tensorData![1] is int) {
        DateTime startTime;
        startTime = DateTime.now();

        List<int> tensorData =
            this._tensorData!.map((num) => num.toInt()).toList();
        Int32List tensorList = Int32List.fromList(tensorData);

        List<int> shapeData = this._shape!.map((num) => num.toInt()).toList();
        Int32List shapeList = Int32List.fromList(shapeData);

        List<int> targetShapeData =
            targetShape!.map((num) => num.toInt()).toList();
        Int32List targetShapeList = Int32List.fromList(targetShapeData);

        ffi.Pointer<ffi.Int32> tensorPtr =
            malloc.allocate(tensorList.lengthInBytes);
        ffi.Pointer<ffi.Int32> shapePtr =
            malloc.allocate(shapeList.lengthInBytes);
        ffi.Pointer<ffi.Int32> targetShapePtr =
            malloc.allocate(targetShapeList.lengthInBytes);

        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensorLen =
            malloc.allocate(8);
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);

        tensorPtr.asTypedList(tensorList.lengthInBytes).setAll(0, tensorList);
        shapePtr.asTypedList(shapeList.lengthInBytes).setAll(0, shapeList);
        targetShapePtr
            .asTypedList(targetShapeList.lengthInBytes)
            .setAll(0, targetShapeList);
        TensorData rawtensorData = _reshape_Int(
          this._tensorData!.length,
          tensorPtr,
          this._shape!.length,
          shapePtr,
          targetShape!.length,
          targetShapePtr,
          outputTensorLen,
          outputShapeLen,
        );
        //

        int tensorLen = rawtensorData.tensorbytesLen;
        int shapeLen = rawtensorData.shapebytesLen;
        dev.log('$shapeLen');
// 检索张量和形状数据
        ffi.Pointer<ffi.Int32> cppTensorPtr = outputTensorLen.value;
        Int32List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
        ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
        Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
        Tensor outputTensor =
            Tensor(rawOutputTensor.toList(), rawOutputShape.toList());


        if (outputTensor != null) {
          return outputTensor;
        } else {
          throw Exception("output Tensor is empty");
        }
      } else if (this._tensorData![1] is double) {
        DateTime startTime;
        startTime = DateTime.now();

        List<double> tensorData =
            this._tensorData!.map((num) => num.toDouble()).toList();
        Float64List tensorList = Float64List.fromList(tensorData);
        List<int> shapeData = this._shape!.map((num) => num.toInt()).toList();
        Int32List shapeList = Int32List.fromList(shapeData);
        List<int> targetShapeData =
            targetShape!.map((num) => num.toInt()).toList();
        Int32List targetShapeList = Int32List.fromList(targetShapeData);

        ffi.Pointer<ffi.Double> tensorPtr =
            malloc.allocate(tensorList.lengthInBytes);
        ffi.Pointer<ffi.Int32> shapePtr =
            malloc.allocate(shapeList.lengthInBytes);
        ffi.Pointer<ffi.Int32> targetShapePtr =
            malloc.allocate(targetShapeList.lengthInBytes);

        ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensorLen =
            malloc.allocate(8);
        ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
        //copy the image data into the memory heap we just allocated
        tensorPtr.asTypedList(tensorList.lengthInBytes).setAll(0, tensorList);
        shapePtr.asTypedList(shapeList.lengthInBytes).setAll(0, shapeList);
        targetShapePtr
            .asTypedList(targetShapeList.lengthInBytes)
            .setAll(0, targetShapeList);
        //c++ image processing
        //image in memory heap -> processing... -> processed image in memory heap
        TensorData rawtensorData = _reshape_Float(
          this._tensorData!.length,
          tensorPtr,
          this._shape!.length,
          shapePtr,
          targetShape!.length,
          targetShapePtr,
          outputTensorLen,
          outputShapeLen,
        );

        int tensorLen = rawtensorData.tensorbytesLen;
        int shapeLen = rawtensorData.shapebytesLen;
        dev.log('$shapeLen');
// 检索张量和形状数据
        ffi.Pointer<ffi.Double> cppTensorPtr = outputTensorLen.value;
        Float64List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
        ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
        Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
        Tensor outputTensor =
            Tensor(rawOutputTensor.toList(), rawOutputShape.toList());
        DateTime endTime = DateTime.now();
        Duration elapsedTime = endTime.difference(startTime);
        dev.log(
            'Function reshape took ${elapsedTime.inMilliseconds} milliseconds to execute.');

        return outputTensor;
      }
      // 如果Tensor数据不是预期类型
      else {
        throw Exception("Tensor数据不是预期的类型");
      }
    } else {
      // 如果Tensor数据或形状为空
      throw Exception("input Tensor is empty");
    }
  }

  Tensor clone() {
    if (this.tensorData != null && this.shape != null) {
      Tensor newTensor = Tensor(this.tensorData, this.shape);
      return newTensor;
    } else {
      throw Exception("Tensor is empty");
    }
  }

  void _processList(List<num> list) {
    // 处理列表的函数
    if (list is List<int>) {
      // 处理整数列表的操作
      print('Processing List<int>: $list');
    } else if (list is List<double>) {
      // 处理浮点数列表的操作
      print('Processing List<double>: $list');
    } else {
      // 处理其他类型的列表，如果有的话
      print('Processing List<num>: $list');
    }
    void processList() {
      if (_tensorData is List<int>) {
        // 如果成员类型是 List<int>，调用处理整数列表的函数
        _processList(_tensorData!);
      } else if (_tensorData is List<double>) {
        // 如果成员类型是 List<double>，调用处理浮点数列表的函数
        _processList(_tensorData!);
      }
    }
  }
}

Tensor transpose(Tensor tensor, int dim1, int dim2) {
  if (tensor._tensorData != null &&
      tensor._shape != null &&
      dim1 < tensor._shape!.length &&
      dim2 < tensor._shape!.length) {
    if (tensor._tensorData![1] is int) {
      DateTime startTime;
      startTime = DateTime.now();

      List<int> tensorData =
          tensor._tensorData!.map((num) => num.toInt()).toList();
      Int32List tensorList = Int32List.fromList(tensorData);
      List<int> shapeData = tensor._shape!.map((num) => num.toInt()).toList();
      Int32List shapeList = Int32List.fromList(shapeData);
      ffi.Pointer<ffi.Int32> tensorPtr =
          malloc.allocate(tensorList.lengthInBytes);
      ffi.Pointer<ffi.Int32> shapePtr =
          malloc.allocate(shapeList.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensorPtr.asTypedList(tensorList.lengthInBytes).setAll(0, tensorList);
      shapePtr.asTypedList(shapeList.lengthInBytes).setAll(0, shapeList);
      
      TensorData rawtensorData = _transpose_Int(
          tensor._tensorData!.length,
          tensorPtr,
          tensor._shape!.length,
          shapePtr,
          outputTensorLen,
          outputShapeLen,
          dim1,
          dim2);
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Int32> cppTensorPtr = outputTensorLen.value;
      Int32List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());

      dev.log('${rawtensorData.tensorbytesLen}');
      dev.log('${rawtensorData.shapebytesLen}');
      dev.log('${outputTensor._tensorData}');
      dev.log('${outputTensor._shape}');
      if (outputTensor != null) {
        return outputTensor;
      } else {
        throw Exception("output Tensor is empty");
      }
    } else if (tensor._tensorData![1] is double) {
      DateTime startTime;
      startTime = DateTime.now();

      List<double> tensorData =
          tensor._tensorData!.map((num) => num.toDouble()).toList();
      Float64List tensorList = Float64List.fromList(tensorData);
      List<int> shapeData = tensor._shape!.map((num) => num.toInt()).toList();
      Int32List shapeList = Int32List.fromList(shapeData);
      ffi.Pointer<ffi.Double> tensorPtr =
          malloc.allocate(tensorList.lengthInBytes);
      ffi.Pointer<ffi.Int32> shapePtr =
          malloc.allocate(shapeList.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensorPtr.asTypedList(tensorList.lengthInBytes).setAll(0, tensorList);
      shapePtr.asTypedList(shapeList.lengthInBytes).setAll(0, shapeList);
   
      TensorData rawtensorData = _transpose_Float(
          tensor._tensorData!.length,
          tensorPtr,
          tensor._shape!.length,
          shapePtr,
          outputTensorLen,
          outputShapeLen,
          dim1,
          dim2);
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Double> cppTensorPtr = outputTensorLen.value;
      Float64List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());
      DateTime endTime = DateTime.now();
      Duration elapsedTime = endTime.difference(startTime);
      dev.log(
          'Function transpose took ${elapsedTime.inMilliseconds} milliseconds to execute.');

      return outputTensor;
    }
    // 如果Tensor数据不是预期类型
    else {
      throw Exception("Tensor数据不是预期的类型");
    }
  } else {
    // 如果Tensor数据或形状为空
    throw Exception("input Tensor is empty");
  }
}

Tensor add(Tensor tensor1,Tensor tensor2,[int alpha=1]) {
  if (tensor1._tensorData != null &&
      tensor1._shape != null &&
      tensor1._tensorData != null &&
      tensor1._shape != null
      ) {
    if (tensor1._tensorData![1] is int&&tensor2._tensorData![1] is int) {
      DateTime startTime;
      startTime = DateTime.now();

      List<int> tensor1Data =
          tensor1._tensorData!.map((num) => num.toInt()).toList();
      Int32List tensor1List = Int32List.fromList(tensor1Data);
      List<int> shape1Data = tensor1._shape!.map((num) => num.toInt()).toList();
      Int32List shape1List = Int32List.fromList(shape1Data);
      ffi.Pointer<ffi.Int32> tensor1Ptr =
          malloc.allocate(tensor1List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape1Ptr =
          malloc.allocate(shape1List.lengthInBytes);

      List<int> tensor2Data =
          tensor2._tensorData!.map((num) => num.toInt()).toList();
      Int32List tensor2List = Int32List.fromList(tensor2Data);
      List<int> shape2Data = tensor2._shape!.map((num) => num.toInt()).toList();
      Int32List shape2List = Int32List.fromList(shape2Data);
      ffi.Pointer<ffi.Int32> tensor2Ptr =
          malloc.allocate(tensor2List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape2Ptr =
          malloc.allocate(shape2List.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensor1Ptr.asTypedList(tensor1List.lengthInBytes).setAll(0, tensor1List);
      shape1Ptr.asTypedList(shape1List.lengthInBytes).setAll(0, shape1List);

      tensor2Ptr.asTypedList(tensor2List.lengthInBytes).setAll(0, tensor2List);
      shape2Ptr.asTypedList(shape2List.lengthInBytes).setAll(0, shape2List);
      
      TensorData rawtensorData = _add_Int(
          tensor1._tensorData!.length,
          tensor1Ptr,
          tensor1._shape!.length,
          shape1Ptr,
          tensor2._tensorData!.length,
          tensor2Ptr,
          tensor2._shape!.length,
          shape2Ptr,
          outputTensorLen,
          outputShapeLen,
          alpha
          );
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Int32> cppTensorPtr = outputTensorLen.value;
      Int32List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());

      dev.log('${rawtensorData.tensorbytesLen}');
      dev.log('${rawtensorData.shapebytesLen}');
      dev.log('${outputTensor._tensorData}');
      dev.log('${outputTensor._shape}');
      if (outputTensor != null) {
        return outputTensor;
      } else {
        throw Exception("output Tensor is empty");
      }
    } else if (tensor1._tensorData![1] is double && tensor2._tensorData![1] is double) {
      DateTime startTime;
      startTime = DateTime.now();

      List<double> tensor1Data =
          tensor1._tensorData!.map((num) => num.toDouble()).toList();
      Float64List tensor1List = Float64List.fromList(tensor1Data);
      List<int> shape1Data = tensor1._shape!.map((num) => num.toInt()).toList();
      Int32List shape1List = Int32List.fromList(shape1Data);
      ffi.Pointer<ffi.Double> tensor1Ptr =
          malloc.allocate(tensor1List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape1Ptr =
          malloc.allocate(shape1List.lengthInBytes);

       List<double> tensor2Data =
          tensor2._tensorData!.map((num) => num.toDouble()).toList();
      Float64List tensor2List = Float64List.fromList(tensor2Data);
      List<int> shape2Data = tensor2._shape!.map((num) => num.toInt()).toList();
      Int32List shape2List = Int32List.fromList(shape2Data);
      ffi.Pointer<ffi.Double> tensor2Ptr =
          malloc.allocate(tensor2List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape2Ptr =
          malloc.allocate(shape2List.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensor1Ptr.asTypedList(tensor1List.lengthInBytes).setAll(0, tensor1List);
      shape1Ptr.asTypedList(shape1List.lengthInBytes).setAll(0, shape1List);

       tensor2Ptr.asTypedList(tensor2List.lengthInBytes).setAll(0, tensor2List);
      shape2Ptr.asTypedList(shape2List.lengthInBytes).setAll(0, shape2List);
     
      TensorData rawtensorData = _add_Float(
          tensor1._tensorData!.length,
          tensor1Ptr,
          tensor1._shape!.length,
          shape1Ptr,
           tensor2._tensorData!.length,
          tensor2Ptr,
          tensor2._shape!.length,
          shape2Ptr,
          outputTensorLen,
          outputShapeLen,
          alpha
          );
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Double> cppTensorPtr = outputTensorLen.value;
      Float64List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());
      DateTime endTime = DateTime.now();
      Duration elapsedTime = endTime.difference(startTime);
      dev.log(
          'Function add took ${elapsedTime.inMilliseconds} milliseconds to execute.');

      return outputTensor;
    }
    // 如果Tensor数据不是预期类型
    else {
      throw Exception("Tensor数据不是预期的类型");
    }
  } else {
    // 如果Tensor数据或形状为空
    throw Exception("input Tensor is empty");
  }
}





Tensor sub(Tensor tensor1,Tensor tensor2,[int alpha=1]) {
  if (tensor1._tensorData != null &&
      tensor1._shape != null &&
      tensor1._tensorData != null &&
      tensor1._shape != null
      ) {
    if (tensor1._tensorData![1] is int&&tensor2._tensorData![1] is int) {
      DateTime startTime;
      startTime = DateTime.now();

      List<int> tensor1Data =
          tensor1._tensorData!.map((num) => num.toInt()).toList();
      Int32List tensor1List = Int32List.fromList(tensor1Data);
      List<int> shape1Data = tensor1._shape!.map((num) => num.toInt()).toList();
      Int32List shape1List = Int32List.fromList(shape1Data);
      ffi.Pointer<ffi.Int32> tensor1Ptr =
          malloc.allocate(tensor1List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape1Ptr =
          malloc.allocate(shape1List.lengthInBytes);

      List<int> tensor2Data =
          tensor2._tensorData!.map((num) => num.toInt()).toList();
      Int32List tensor2List = Int32List.fromList(tensor2Data);
      List<int> shape2Data = tensor2._shape!.map((num) => num.toInt()).toList();
      Int32List shape2List = Int32List.fromList(shape2Data);
      ffi.Pointer<ffi.Int32> tensor2Ptr =
          malloc.allocate(tensor2List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape2Ptr =
          malloc.allocate(shape2List.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      //copy the image data into the memory heap we just allocated
      tensor1Ptr.asTypedList(tensor1List.lengthInBytes).setAll(0, tensor1List);
      shape1Ptr.asTypedList(shape1List.lengthInBytes).setAll(0, shape1List);

      tensor2Ptr.asTypedList(tensor2List.lengthInBytes).setAll(0, tensor2List);
      shape2Ptr.asTypedList(shape2List.lengthInBytes).setAll(0, shape2List);
      
      TensorData rawtensorData = _sub_Int(
          tensor1._tensorData!.length,
          tensor1Ptr,
          tensor1._shape!.length,
          shape1Ptr,
          tensor2._tensorData!.length,
          tensor2Ptr,
          tensor2._shape!.length,
          shape2Ptr,
          outputTensorLen,
          outputShapeLen,
          alpha
          );
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Int32> cppTensorPtr = outputTensorLen.value;
      Int32List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());

      dev.log('${rawtensorData.tensorbytesLen}');
      dev.log('${rawtensorData.shapebytesLen}');
      dev.log('${outputTensor._tensorData}');
      dev.log('${outputTensor._shape}');
      if (outputTensor != null) {
        return outputTensor;
      } else {
        throw Exception("output Tensor is empty");
      }
    } else if (tensor1._tensorData![1] is double && tensor2._tensorData![1] is double) {
      DateTime startTime;
      startTime = DateTime.now();

      List<double> tensor1Data =
          tensor1._tensorData!.map((num) => num.toDouble()).toList();
      Float64List tensor1List = Float64List.fromList(tensor1Data);
      List<int> shape1Data = tensor1._shape!.map((num) => num.toInt()).toList();
      Int32List shape1List = Int32List.fromList(shape1Data);
      ffi.Pointer<ffi.Double> tensor1Ptr =
          malloc.allocate(tensor1List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape1Ptr =
          malloc.allocate(shape1List.lengthInBytes);

       List<double> tensor2Data =
          tensor2._tensorData!.map((num) => num.toDouble()).toList();
      Float64List tensor2List = Float64List.fromList(tensor2Data);
      List<int> shape2Data = tensor2._shape!.map((num) => num.toInt()).toList();
      Int32List shape2List = Int32List.fromList(shape2Data);
      ffi.Pointer<ffi.Double> tensor2Ptr =
          malloc.allocate(tensor2List.lengthInBytes);
      ffi.Pointer<ffi.Int32> shape2Ptr =
          malloc.allocate(shape2List.lengthInBytes);

      ffi.Pointer<ffi.Pointer<ffi.Double>> outputTensorLen = malloc.allocate(8);
      ffi.Pointer<ffi.Pointer<ffi.Int32>> outputShapeLen = malloc.allocate(8);
      
      tensor1Ptr.asTypedList(tensor1List.lengthInBytes).setAll(0, tensor1List);
      shape1Ptr.asTypedList(shape1List.lengthInBytes).setAll(0, shape1List);

       tensor2Ptr.asTypedList(tensor2List.lengthInBytes).setAll(0, tensor2List);
      shape2Ptr.asTypedList(shape2List.lengthInBytes).setAll(0, shape2List);
      
      TensorData rawtensorData = _sub_Float(
          tensor1._tensorData!.length,
          tensor1Ptr,
          tensor1._shape!.length,
          shape1Ptr,
           tensor2._tensorData!.length,
          tensor2Ptr,
          tensor2._shape!.length,
          shape2Ptr,
          outputTensorLen,
          outputShapeLen,
          alpha
          );
      //

      int tensorLen = rawtensorData.tensorbytesLen;
      int shapeLen = rawtensorData.shapebytesLen;
      dev.log('$shapeLen');
// 检索张量和形状数据
      ffi.Pointer<ffi.Double> cppTensorPtr = outputTensorLen.value;
      Float64List rawOutputTensor = cppTensorPtr.asTypedList(tensorLen);
      ffi.Pointer<ffi.Int32> cppShapePtr = outputShapeLen.value; // 正确的变量用于形状
      Int32List rawOutputShape = cppShapePtr.asTypedList(shapeLen);
      Tensor outputTensor =
          Tensor(rawOutputTensor.toList(), rawOutputShape.toList());
      DateTime endTime = DateTime.now();
      Duration elapsedTime = endTime.difference(startTime);
      dev.log(
          'Function sub took ${elapsedTime.inMilliseconds} milliseconds to execute.');

      return outputTensor;
    }
    // 如果Tensor数据不是预期类型
    else {
      throw Exception("Tensor数据不是预期的类型");
    }
  } else {
    // 如果Tensor数据或形状为空
    throw Exception("input Tensor is empty");
  }
}


Tensor createTensorFromList(dynamic list, [List<int>? sizes]) {
  List<num> flatList = [];
  List<int> sizes = [];
  bool isFirstElement = true; // 用于标记是否是每个维度的第一个元素

  // 递归展平列表，并记录维度大小
  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElement) {
        // 只在遇到新维度的第一个元素时记录大小
        sizes.add(element.length);
        isFirstElement = false;
      }
      for (var subElement in element) {
        flatten(subElement, depth + 1);
      }
    } else if (element is num) {
      flatList.add(element);
      isFirstElement = true; // 遇到非列表元素，重置标记，为下一维度准备
    }
  }

  flatten(list, 0);
  Tensor outputTensor = Tensor(flatList, sizes);
  return outputTensor;
}

List<dynamic> createListFromTensor(Tensor tensor) {
  // 克隆传入的 tensor 以避免修改原始数据
  Tensor inputTensor = tensor.clone();
  List<num> flatList = List<num>.from(inputTensor._tensorData!);
  List<int> shape = List<int>.from(inputTensor._shape!);

  // 计算每个维度的步长
  List<int> strides = List<int>.filled(shape.length, 1);
  for (int i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  List<dynamic> buildList(int dimension, int offset) {
    if (dimension == shape.length - 1) {
      return flatList.sublist(offset, offset + shape[dimension]);
    }

    List<dynamic> result = [];
    for (int i = 0; i < shape[dimension]; i++) {
      result.add(buildList(dimension + 1, offset + i * strides[dimension]));
    }
    return result;
  }

  return buildList(0, 0);
}
