import 'dart:ffi';
import 'dart:io';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'dart:developer' as dev;
import 'constants.dart';

final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libpytorch_dart.so')
    : DynamicLibrary.process();

final Pointer<Utf8> Function(Pointer<Int64> size, int length, int requiresGrad,int dtype,
        Pointer<Pointer<Void>> result) _empty =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Int64> size,
                    Int64 length,
                    Int64 requiresGrad,
                    Int8 dtype,
                    Pointer<Pointer<Void>> result)>>('Empty')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Int64> size, int length, int requiresGrad,int dtype,
        Pointer<Pointer<Void>> result) _ones =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Int64> size, Int64 length,
                    Int64 requiresGrad,Int8 dtype, Pointer<Pointer<Void>> result)>>('Ones')
        .asFunction();
final Pointer<Utf8> Function(
        int n, int m, int requiresGrad,int dtype, Pointer<Pointer<Void>> result) _eye =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Int64 n, Int64 m, Int64 requiresGrad,Int8 dtype,
                    Pointer<Pointer<Void>> result)>>('Eye')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Int64> size, int length, double value,
        int requiresGrad, Pointer<Pointer<Void>> result) _full =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Int64> size,
                    Int64 length,
                    Float value,
                    Int64 requiresGrad,
                    Pointer<Pointer<Void>> result)>>('Full')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void>) _print = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void>)>>(
        'Tensor_Print')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>> result)
    _detach = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> tensor,
                    Pointer<Pointer<Void>> result)>>('Tensor_Detach')
        .asFunction();
final Pointer<Utf8> Function(Pointer<Void>, Pointer<Int64>,int,Pointer<Pointer<Void>>) _index = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(
                Pointer<Void>, Pointer<Int64>,Int64,Pointer<Pointer<Void>>)>>('Tensor_Index')
    .asFunction();
final Pointer<Utf8> Function(Pointer<Void>, Pointer<Int64>) _shape = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(
                Pointer<Void>, Pointer<Int64>)>>('Tensor_Shape')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Int64>) _dim = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(
                Pointer<Void>, Pointer<Int64>)>>('Tensor_Dim')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Int8>) _dtype = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(
                Pointer<Void>, Pointer<Int8>)>>('Tensor_Dtype')
    .asFunction();

final Pointer<Utf8> Function(
        Pointer<Void> data,
        int dtype,
        Pointer<Int64> sizes_data,
        int sizes_data_len,
        Pointer<Pointer<Void>> result) _from_blob =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Void> data,
                    Int8 dtype,
                    Pointer<Int64> sizes_data,
                    Int64 sizes_data_len,
                    Pointer<Pointer<Void>> result)>>('Tensor_FromBlob')
        .asFunction();

        final Pointer<Utf8> Function(
        Pointer<Void> a,
        int dtype,
        Pointer<Int> result) _toList_Int =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Void> a,
                    Int8 dtype,
                    Pointer<Int> result)>>('Tensor_ToArray_Int')
        .asFunction();
        
        final Pointer<Utf8> Function(
        Pointer<Void> a,
        int dtype,
        Pointer<Float> result) _toList_Float =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Void> a,
                    Int8 dtype,
                    Pointer<Float> result)>>('Tensor_ToArray_Float')
        .asFunction();

        
        final Pointer<Utf8> Function(
        Pointer<Void> a,
        int dtype,
        Pointer<Double> result) _toList_Float64 =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Void> a,
                    Int8 dtype,
                    Pointer<Double> result)>>('Tensor_ToArray_Float64')
        .asFunction();

final Pointer<Utf8> Function(double start, double end, double step,
        int requiresGrad, Pointer<Pointer<Void>> result) _arange =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Float start,
                    Float end,
                    Float step,
                    Int64 requiresGrad,
                    Pointer<Pointer<Void>> result)>>('Arange')
        .asFunction();

final Pointer<Utf8> Function(double start, double end, int steps,
        int requiresGrad, Pointer<Pointer<Void>> result) _linspace =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Float start,
                    Float end,
                    Int64 steps,
                    Int64 requiresGrad,
                    Pointer<Pointer<Void>> result)>>('Linspace')
        .asFunction();

final Pointer<Utf8> Function(double start, double end, int steps, double base,
        int requiresGrad, Pointer<Pointer<Void>> result) _logspace =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Float start,
                    Float end,
                    Int64 steps,
                    Double base,
                    Int64 requiresGrad,
                    Pointer<Pointer<Void>> result)>>('Logspace')
        .asFunction();

final Pointer<Utf8> Function(
        Pointer<Void> a, Pointer<Void> b, Pointer<Int64> result) _equal =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> b,
                    Pointer<Int64> result)>>('Equal')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void> a, Pointer<Pointer<Void>> result)
    _sum = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Void> a, Pointer<Pointer<Void>> result)>>('Sum')
        .asFunction();

final Pointer<Utf8> Function(
        Pointer<Void> a, int dim, int keepDim, Pointer<Pointer<Void>> result)
    _sumByDim = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> a, Int64 dim, Int8 keepDim,
                    Pointer<Pointer<Void>> result)>>('SumByDim')
        .asFunction();

final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  Pointer<Pointer<Void>> result,
) _mm = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Pointer<Pointer<Void>> result)>>('MM')
    .asFunction();

    final Pointer<Utf8> Function(Pointer<Void>,int dim0,int dim1, Pointer<Pointer<Void>> result)
    _transpose = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> tensor,Int64 dim0,Int64 dim1,
                    Pointer<Pointer<Void>> result)>>('Transpose')
        .asFunction();

        final Pointer<Utf8> Function(Pointer<Void>,Pointer<Int64>,int dim_size, Pointer<Pointer<Void>> result)
    _permute = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> tensor,Pointer<Int64>,Int64 dim_size,
                    Pointer<Pointer<Void>> result)>>('Permute')
        .asFunction();

final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  double alpha,
  Pointer<Pointer<Void>> result,
) _add = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Float alpha, Pointer<Pointer<Void>> result)>>('Add')
    .asFunction();

// Add_
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  double alpha,
  Pointer<Pointer<Void>> result,
) _add_ = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Float alpha, Pointer<Pointer<Void>> result)>>('Add_')
    .asFunction();

// Sub
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  double alpha,
  Pointer<Pointer<Void>> result,
) _sub = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Float alpha, Pointer<Pointer<Void>> result)>>('Sub')
    .asFunction();

// Sub_
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  double alpha,
  Pointer<Pointer<Void>> result,
) _sub_ = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Float alpha, Pointer<Pointer<Void>> result)>>('Sub_')
    .asFunction();

// Mul
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  Pointer<Pointer<Void>> result,
) _mul = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Pointer<Pointer<Void>> result)>>('Mul')
    .asFunction();

// Mul_
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  Pointer<Pointer<Void>> result,
) _mul_ = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Pointer<Pointer<Void>> result)>>('Mul_')
    .asFunction();

// Div
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  Pointer<Pointer<Void>> result,
) _div = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Pointer<Pointer<Void>> result)>>('Div')
    .asFunction();

// Div_
final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Void> other,
  Pointer<Pointer<Void>> result,
) _div_ = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Void> other,
                Pointer<Pointer<Void>> result)>>('Div_')
    .asFunction();

final Pointer<Utf8> Function(
  Pointer<Void> a,
  Pointer<Utf8> path
) _save = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Void> a, Pointer<Utf8>)>>('Tensor_Save')
    .asFunction();
final Pointer<Utf8> Function(
  Pointer<Utf8> path,
  Pointer<Pointer<Void>> a
) _load = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Utf8> Function(Pointer<Utf8> path,Pointer<Pointer<Void>> a)>>('Tensor_Load')
    .asFunction();


final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>> result) _relu = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Pointer<Void>> result)>>('Relu')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, double, Pointer<Pointer<Void>> result) _leakyRelu = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Double negative_slope, Pointer<Pointer<Void>> result)>>('LeakyRelu')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>> result) _tanh = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Pointer<Void>> result)>>('Tanh')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>> result) _sigmoid = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Pointer<Void>> result)>>('Sigmoid')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Void>, Pointer<Int64> result) _allClose = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensorA, Pointer<Void> tensorB, Pointer<Int64> result)>>('AllClose')
    .asFunction();


final Pointer<Utf8> Function(Pointer<Void>, int,int, Pointer<Pointer<Void>> result) _flatten = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Int64 startDim, Int64 endDim, Pointer<Pointer<Void>> result)>>('Flatten')
    .asFunction();

  final Pointer<Utf8> Function(Pointer<Void>, int,int,int,int, Pointer<Pointer<Void>>, Pointer<Pointer<Void>>) _topK = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Int64 k, Int64 dim, Int8 largest, Int8 sorted, Pointer<Pointer<Void>> values, Pointer<Pointer<Void>> indices)>>('TopK')
    .asFunction();

// 类定义


class Tensor {
  Pointer<Void> _tensorPtr;

  Tensor._internal(this._tensorPtr);
  
  Tensor operator +(dynamic b){
    double alpha=1;
    if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(this._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
    

  }

   Tensor operator -(dynamic b){
    double alpha=1;
    if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(this._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
    

  }

  Tensor operator*(dynamic b){
 if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(this._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(this._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(this._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}

Tensor operator/(dynamic b){
 if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(this._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(this._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(this._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}

 Tensor operator[](int index_num){

  return index([index_num]);
 }




  @override
  String toString() {
    var stringPtr = _print(_tensorPtr);
    final string = stringPtr.cast<Utf8>().toDartString();
    return string;
  }

  Tensor detach() {
    final resultTensorPtr = calloc<Pointer<Void>>();

    final errorMsg = _detach(_tensorPtr, resultTensorPtr);

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr); // 释放结果指针

    return tensor;
  }

  int dim() {
    final dimPtr = calloc<Int64>();

    final errorMsg = _dim(_tensorPtr, dimPtr);

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }

    Int64List rawDim = dimPtr.asTypedList(1);
    int dim = rawDim[0];
    calloc.free(dimPtr); // 释放结果指针

    return dim;
  }
 int dtype() {
    final dtypePtr = calloc<Int8>();

    final errorMsg = _dtype(_tensorPtr, dtypePtr);

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }

    Int8List rawDtype = dtypePtr.asTypedList(1);
    int dtype = rawDtype[0];
    calloc.free(dtypePtr); // 释放结果指针

    return dtype;
  }


  List<int> shape() {
    final shapePtr = calloc<Int64>(dim());

    final errorMsg = _shape(_tensorPtr, shapePtr);

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }

    Int64List rawShape = shapePtr.asTypedList(dim());
    
    List<int> shape = List<int>.from(rawShape);
    
    return shape;
  }

  List<int> size() {
    final shapePtr = calloc<Int64>(dim());

    final errorMsg = _shape(_tensorPtr, shapePtr);

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }

    Int64List rawShape = shapePtr.asTypedList(dim());
    
    List<int> shape = List<int>.from(rawShape);
    
    return shape;
  }

  
  void add_(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(this._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);//maybe change to torch.full()
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
  }

  void sub_(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(this._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);//maybe change to torch.full()
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(this._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
  }

 void mul_(dynamic b){
 if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(this._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(this._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

       this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(this._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

       this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}


 void div_(dynamic b){
 if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(this._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(this._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

       this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(this._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

       this._tensorPtr = resultTensorPtr.value;
      calloc.free(resultTensorPtr);
      
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}

  Tensor index(List<int> index_list)
  {final Pointer<Int64> indexPointer = malloc<Int64>(index_list.length);
     final Int64List indexList = indexPointer.asTypedList(index_list.length);
    indexList.setAll(0, index_list);
final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _index(this._tensorPtr,indexPointer,index_list.length, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;

  }
  



Tensor transpose(int dim0,int dim1) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _transpose(this._tensorPtr,dim0,dim1,resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}





   Tensor permute(List<int> permute_list)
  {final Pointer<Int64> permutePointer = malloc<Int64>(permute_list.length);
     final Int64List permuteList = permutePointer.asTypedList(permute_list.length);
    permuteList.setAll(0, permute_list);
final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _permute(this._tensorPtr,permutePointer,permute_list.length, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;

  }  
  


List<dynamic> toList() {
 var Dtype=this.dtype();
  var tensorLength=this.shape().reduce((value, element) => value * element);
  List<int> tensorShape = this.shape();
  if (Dtype==int32) {
    
    // 获取该数组的指针
    

    // 创建 sizes 数组的指针
   

    final resultListPtr = calloc<Int>(tensorLength);
    
    final errorMsg = _toList_Int(
        this._tensorPtr, Dtype,  resultListPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
   final Pointer<Int32> dataPointer = resultListPtr.cast<Int32>();

  final Int32List flatList = dataPointer.asTypedList(tensorLength);
  

  List<int> strides = List<int>.filled(tensorShape.length, 1);
  for (int i = tensorShape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * tensorShape[i + 1];
  }

  List<dynamic> buildList(int dimension, int offset) {
    if (dimension == tensorShape.length - 1) {
      return flatList.sublist(offset, offset + tensorShape[dimension]);
    }

    List<dynamic> result = [];
    for (int i = 0; i < tensorShape[dimension]; i++) {
      result.add(buildList(dimension + 1, offset + i * strides[dimension]));
    }
    return result;
  }

  return buildList(0, 0);

  } else if (Dtype == float32) {
     
    // 获取该数组的指针
    

    // 创建 sizes 数组的指针
   

    final resultListPtr = calloc<Float>(tensorLength);
    
    final errorMsg = _toList_Float(
        this._tensorPtr, Dtype,  resultListPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
   
  final Pointer<Float> dataPointer = resultListPtr.cast<Float>();
  final Float32List flatList = dataPointer.asTypedList(tensorLength);
    List<int> strides = List<int>.filled(tensorShape.length, 1);
  for (int i = tensorShape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * tensorShape[i + 1];
  }

  List<dynamic> buildList(int dimension, int offset) {
    if (dimension == tensorShape.length - 1) {
      return flatList.sublist(offset, offset + tensorShape[dimension]);
    }

    List<dynamic> result = [];
    for (int i = 0; i < tensorShape[dimension]; i++) {
      result.add(buildList(dimension + 1, offset + i * strides[dimension]));
    }
    return result;
  }

  return buildList(0, 0);
  
  }
  else if (Dtype == float64) {
     
    // 获取该数组的指针
    

    // 创建 sizes 数组的指针
   

    final resultListPtr = calloc<Double>(tensorLength);
    
    final errorMsg = _toList_Float64(
        this._tensorPtr, Dtype,  resultListPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
   final Pointer<Double> dataPointer = resultListPtr.cast<Double>();

  final Float64List flatList = dataPointer.asTypedList(tensorLength);
   List<int> strides = List<int>.filled(tensorShape.length, 1);
  for (int i = tensorShape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * tensorShape[i + 1];
  }

  List<dynamic> buildList(int dimension, int offset) {
    if (dimension == tensorShape.length - 1) {
      return flatList.sublist(offset, offset + tensorShape[dimension]);
    }

    List<dynamic> result = [];
    for (int i = 0; i < tensorShape[dimension]; i++) {
      result.add(buildList(dimension + 1, offset + i * strides[dimension]));
    }
    return result;
  }

  return buildList(0, 0);
  }
  else {
    throw Exception("wrong type");
  }

  // 使用完毕后释放指针
}





}


Tensor relu(Tensor a) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _relu(a._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}
Tensor leakyRelu(Tensor a,double negative_slope) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _leakyRelu(a._tensorPtr, negative_slope, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}

Tensor tanh(Tensor a) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _tanh(a._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}

Tensor sigmoid(Tensor a) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _sigmoid(a._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}

bool allClose(Tensor a,Tensor other) {
  final resultValue = calloc<Int64>();
  final errorMsg = _allClose(a._tensorPtr, other._tensorPtr, resultValue);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final bool close = resultValue.value == 1;
  calloc.free(resultValue);
  
  return close;
}

Tensor flatten(Tensor a,int startDim, int endDim) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _flatten(a._tensorPtr, startDim, endDim, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}


Map<String, Tensor> topk(Tensor a,int k, int dim, {bool largest = true, bool sorted = true}) {
  final valuesTensorPtr = calloc<Pointer<Void>>();
  final indicesTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _topK(a._tensorPtr, k, dim, largest ? 1 : 0, sorted ? 1 : 0, valuesTensorPtr, indicesTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final valuesTensor = Tensor._internal(valuesTensorPtr.value);
  final indicesTensor = Tensor._internal(indicesTensorPtr.value);
  calloc.free(valuesTensorPtr);
  calloc.free(indicesTensorPtr);

  return {'values': valuesTensor, 'indices': indicesTensor};
}

Tensor from_blob(List<num> list, List<int> sizes_data,{int dtype=float32}) {
  if (dtype==int32) {
    var intList = Int32List.fromList(list.cast<int>());
    // 获取该数组的指针
    final Pointer<Int32> dataPointer = malloc<Int32>(intList.length);
    dataPointer
        .asTypedList(intList.length)
        .setRange(0, intList.length, intList);

    // 创建 sizes 数组的指针
    final Pointer<Int64> sizesPointer = malloc<Int64>(sizes_data.length);
    final Int64List sizesList = sizesPointer.asTypedList(sizes_data.length);
    sizesList.setAll(0, sizes_data);

    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _from_blob(
        dataPointer.cast(), dtype, sizesPointer, sizesList.length, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr.value);
    return tensor;
  } else if (dtype == float32) {
    var floatList = Float32List.fromList(list.cast<double>());
    // 获取该数组的指针
    final Pointer<Float> dataPointer = malloc<Float>(floatList.length);
    dataPointer
        .asTypedList(floatList.length)
        .setRange(0, floatList.length, floatList);

    // 创建 sizes 数组的指针
    final Pointer<Int64> sizesPointer = malloc<Int64>(sizes_data.length);
    final Int64List sizesList = sizesPointer.asTypedList(sizes_data.length);
    sizesList.setAll(0, sizes_data);

    // 调用 FFI 函数

    // 调用 FFI 函数
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _from_blob(
        dataPointer.cast(), dtype, sizesPointer, sizesList.length, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr.value);
    return tensor;
  }
  else if (dtype == float64) {
    var floatList = Float64List.fromList(list.cast<double>());
    // 获取该数组的指针
    final Pointer<Double> dataPointer = malloc<Double>(floatList.length);
    dataPointer
        .asTypedList(floatList.length)
        .setRange(0, floatList.length, floatList);

    // 创建 sizes 数组的指针
    final Pointer<Int64> sizesPointer = malloc<Int64>(sizes_data.length);
    final Int64List sizesList = sizesPointer.asTypedList(sizes_data.length);
    sizesList.setAll(0, sizes_data);

    // 调用 FFI 函数

    // 调用 FFI 函数
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _from_blob(
        dataPointer.cast(), dtype, sizesPointer, sizesList.length, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr.value);
    return tensor; 
  }
  else {
    throw Exception("wrong type");
  }

  // 使用完毕后释放指针
}





Tensor empty(List<int> size, {bool requiresGrad = false,int dtype=float32}) {
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _empty(int64Pointer, size.length, requiresGrad ? 1 : 0, dtype,resultTensorPtr);

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr); // 释放结果指针

  return tensor;
}

Tensor ones(List<int> size, {bool requiresGrad = false,int dtype=float32}) {
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _ones(int64Pointer, size.length, requiresGrad ? 1 : 0, dtype,resultTensorPtr);

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr); // 释放结果指针

  return tensor;
}

Tensor full(List<int> size, num values, {bool requiresGrad = false}) {
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _full(int64Pointer, size.length, values.toDouble(),
      requiresGrad ? 1 : 0, resultTensorPtr);

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr); // 释放结果指针

  return tensor;
}

Tensor eye(int n, int m, {bool requiresGrad = false,int dtype=float32}) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _eye(n, m, requiresGrad ? 1 : 0, dtype,resultTensorPtr);

  // 释放原生数组内存

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr); // 释放结果指针

  return tensor;
}

Tensor arange(double start, double end, double step,
    {bool requiresGrad = false}) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _arange(start, end, step, requiresGrad ? 1 : 0, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor linspace(double start, double end, int steps,
    {bool requiresGrad = false}) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _linspace(start, end, steps, requiresGrad ? 1 : 0, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor logspace(double start, double end, int steps, double base,
    {bool requiresGrad = false}) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _logspace(start, end, steps, base, requiresGrad ? 1 : 0, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

bool equal(Tensor a, Tensor b) {
  final resultPtr = calloc<Int64>();
  final errorMsg = _equal(a._tensorPtr, b._tensorPtr, resultPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final result = resultPtr.value != 0;
  calloc.free(resultPtr);

  return result;
}

Tensor mm(Tensor a, Tensor b) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _mm(a._tensorPtr, b._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor transpose(Tensor a,int dim0,int dim1) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _transpose(a._tensorPtr,dim0,dim1,resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

 Tensor permute(Tensor a,List<int> permute_list)
  {final Pointer<Int64> permutePointer = malloc<Int64>(permute_list.length);
     final Int64List permuteList = permutePointer.asTypedList(permute_list.length);
    permuteList.setAll(0, permute_list);
final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _permute(a._tensorPtr,permutePointer,permute_list.length, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;

  }  

Tensor sum(Tensor a) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _sum(a._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor sumByDim(Tensor a, int dim, {bool keepDim = false}) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _sumByDim(a._tensorPtr, dim, keepDim ? 1 : 0, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor add(dynamic a, dynamic b, {double alpha = 1}) {
if(a is num){
  return add(b,a,alpha: alpha);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type");}
}

Tensor sub(dynamic a, dynamic b, {double alpha = 1}) {
 if(a is num){
  return sub(b,a,alpha: alpha);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type");}
}

Tensor mul(dynamic a, dynamic b) {
if(a is num)
{
  return mul(b,a);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul(a._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul(a._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul(a._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type.");}
}

Tensor div(dynamic a, dynamic b) {
  if(a is num)
{
  return div(b,a);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div(a._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div(a._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div(a._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type.");}
}


Tensor add_(dynamic a, dynamic b, {double alpha = 1}) {
  if(a is num){
  return add_(b,a,alpha: alpha);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _add_(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type");}
}

Tensor sub_(dynamic a, dynamic b, {double alpha = 1}) {
 if(a is num){
  return sub_(b,a,alpha: alpha);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _sub_(a._tensorPtr, broadcast._tensorPtr, alpha, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type");}
}

Tensor mul_(dynamic a, dynamic b) {
  if(a is num)
{
  return mul_(b,a);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(a._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(a._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _mul_(a._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type.");}
}

Tensor div_(dynamic a, dynamic b) {
   if(a is num)
{
  return div_(b,a);
}
else if(a is Tensor){
  if (b is Tensor) {
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(a._tensorPtr, b._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      Tensor broadcast = from_blob([b],[1],dtype: int32);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(a._tensorPtr, broadcast._tensorPtr,resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        Tensor broadcast = from_blob([b],[1],dtype: float64);
      final resultTensorPtr = calloc<Pointer<Void>>();
      final errorMsg =
          _div_(a._tensorPtr, broadcast._tensorPtr, resultTensorPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr.value);
      calloc.free(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type.");}
}


void save(Tensor a,String path)
{if(Platform.isWindows || Platform.isLinux){
  final units = utf8.encode(path);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final path_Utf8= result.cast<Utf8>();

final errorMsg = _save(a._tensorPtr,path_Utf8);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);

}
}
else{throw Exception("only support desktop platform");}
}
Tensor load(String path) {
  if(Platform.isWindows || Platform.isLinux){
  final units = utf8.encode(path);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final path_Utf8= result.cast<Utf8>();
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _load(path_Utf8,resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
  }
  else{throw Exception("only support desktop platform");}
}


Tensor IntTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  bool isFirstElement = true;

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElement) {
        sizes.add(element.length);
        isFirstElement = false;
      }
      for (var subElement in element) {
        flatten(subElement, depth + 1);
      }
    } else if (element is num) {
      flatList.add(element);
      isFirstElement = true;
    }
  }

  flatten(list, 0);

  Tensor outputTensor =
      from_blob(flatList.cast<int>(), sizes,dtype: int32);
  return outputTensor;
}

Tensor FloatTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  bool isFirstElement = true;

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElement) {
        sizes.add(element.length);
        isFirstElement = false;
      }
      for (var subElement in element) {
        flatten(subElement, depth + 1);
      }
    } else if (element is num) {
      flatList.add(element);
      isFirstElement = true;
    }
  }

  flatten(list, 0);

  Tensor outputTensor =
      from_blob(flatList.cast<double>(), sizes,dtype:float32);
  return outputTensor;
}

Tensor DoubleTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  bool isFirstElement = true;

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElement) {
        sizes.add(element.length);
        isFirstElement = false;
      }
      for (var subElement in element) {
        flatten(subElement, depth + 1);
      }
    } else if (element is num) {
      flatList.add(element);
      isFirstElement = true;
    }
  }

  flatten(list, 0);

  Tensor outputTensor =
      from_blob(flatList.cast<double>(), sizes,dtype:float64);
  return outputTensor;
}

