import 'dart:ffi';
import 'dart:io';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:pytorch_dart/pytorch_dart.dart';
import 'dart:developer' as dev;
import 'constants.dart';
import "pinnedMemory.dart";
import "scalar.dart";
import "device.dart";
import "GCHandleDeleter.dart";

final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libpytorch_dart.so')
    : DynamicLibrary.process();

final Pointer<Utf8> Function() _get_and_reset_last_err=
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    )>>('THSTorch_get_and_reset_last_err')
        .asFunction();



final Pointer<Void> Function(int value) _int32_to_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Int value)>>('THSTorch_int32_to_scalar')
        .asFunction();

final Pointer<Void> Function(double value) _float32_to_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Float value)>>('THSTorch_float32_to_scalar')
        .asFunction();
final Pointer<Void> Function(double value) _float64_to_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Double value)>>('THSTorch_float64_to_scalar')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_add =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_add')
        .asFunction();


final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_add_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_add_scalar')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_add_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_add_')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_add_scalar_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_add_scalar_')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_sub_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_sub_')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_sub_scalar_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_sub_scalar_')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_sub =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_sub')
        .asFunction();


final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha) Tensor_sub_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Void> alpha)>>('THSTensor_sub_scalar')
        .asFunction();



final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right) Tensor_mul =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right)>>('THSTensor_mul')
        .asFunction();


final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right) Tensor_mul_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right)>>('THSTensor_mul_scalar')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right) Tensor_mul_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right)>>('THSTensor_mul_')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right) Tensor_mul_scalar_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right)>>('THSTensor_mul_scalar_')
        .asFunction();

        final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode) Tensor_div =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode)>>('THSTensor_div')
        .asFunction();


final Pointer<Void> Function(Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode) Tensor_div_scalar =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode)>>('THSTensor_div_scalar')
        .asFunction();

         final void Function(Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode) Tensor_div_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode)>>('THSTensor_div_')
        .asFunction();


final void Function(Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode) Tensor_div_scalar_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left,Pointer<Void> right,Pointer<Utf8> rounding_mode)>>('THSTensor_div_scalar_')
        .asFunction();
final Pointer<Void> Function(Pointer<Int64> sizes,int length, int scalar_type, int device_type,int device_index, bool requires_grad) 
Tensor_empty = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Int64> sizes, Int32 length, Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_empty')
    .asFunction();

final Pointer<Void> Function(Pointer<Int64> sizes,int length, int scalar_type, int device_type,int device_index, bool requires_grad) 
Tensor_ones = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Int64> sizes, Int32 length, Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_ones')
    .asFunction();
final Pointer<Void> Function(int n,int m,int scalar_type, int device_type,int device_index, bool requires_grad) 
Tensor_eye = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Int32 n,Int32 m,  Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_eye')
    .asFunction();

final Pointer<Void> Function(Pointer<Int64> sizes,int length, Pointer<Void> scalar,int scalar_type, int device_type,int device_index, bool requires_grad) 
Tensor_full = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Int64> sizes, Int32 length,Pointer<Void> scalar, Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_full')
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

final Pointer<Void> Function(Pointer<Void> tensor) 
Tensor_detach = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>('THSTensor_detach')
    .asFunction();


    final Pointer<Void> Function(Pointer<Void>, Pointer<Int64>,Pointer<Int64>,Pointer<Int64>,Pointer<Pointer<Void>>,int) Tensor_index = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Void> Function(
                Pointer<Void>, Pointer<Int64>,Pointer<Int64>,Pointer<Int64>,Pointer<Pointer<Void>>,Int64)>>('THSTensor_index')
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

final Pointer<Void> Function(
        Pointer<Void> data,
        Pointer<NativeFunction<DeleterNative>> deleter,
        Pointer<Int64> sizes_data,
        int sizes_data_len,
        int scalar_type,
        int dtype,
        int device_type,
        int device_index,
         bool requires_grad
        ) Tensor_new =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> data,
                     Pointer<NativeFunction<DeleterNative>> deleter,
                     Pointer<Int64> sizes_data,
                    Int64 sizes_data_len,
                     Int8 scalar_type,
                    Int8 dtype,
                    Int device_type,
                    Int device_index,
                    Bool requires_grad
                    )>>('THSTensor_new')
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

final Pointer<Void> Function(Pointer<Void> start, Pointer<Void> end, Pointer<Void> step, int scalar_type, int device_type, int device_index,bool requires_grad) 
Tensor_arange = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> start, Pointer<Void> end, Pointer<Void> step, Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_arange')
    .asFunction();

final Pointer<Void> Function(double start, double end, int steps, int scalar_type, int device_type, int device_index, bool requires_grad) 
Tensor_linspace = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Double start, Double end, Int64 steps, Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_linspace')
    .asFunction();

final Pointer<Void> Function(double start, double end, int steps,double base, int scalar_type, int device_type, int device_index, bool requires_grad) 
Tensor_logspace = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Double start, Double end, Int64 steps,Double base, Int8 scalar_type, Int32 device_type, Int32 device_index, Bool requires_grad)>>('THSTensor_logspace')
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

    final Pointer<Void> Function(Pointer<Void> tensor, int dim1, int dim2) 
Tensor_transpose = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor, Int64 dim1, Int64 dim2)>>('THSTensor_transpose')
    .asFunction();

        final Pointer<Void> Function(Pointer<Void>,Pointer<Int64>,int dim_size)
    Tensor_permute = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor,Pointer<Int64>,Int64 dim_size,
                    )>>('THSTensor_permute')
        .asFunction();


final void Function(
  Pointer<Void> a,
  Pointer<Utf8> path
) Tensor_save = nativeLib
    .lookup<
        NativeFunction<
            Void Function(Pointer<Void> a, Pointer<Utf8>)>>('THSTensor_Save')
    .asFunction();
final Pointer<Void> Function(
  Pointer<Utf8> path,
  
) Tensor_load = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Void> Function(Pointer<Utf8> path)>>('THSTensor_Load')
    .asFunction();


final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>> result) _relu = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Pointer<Void>> result)>>('Relu')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, double, Pointer<Pointer<Void>> result) _leakyRelu = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Double negative_slope, Pointer<Pointer<Void>> result)>>('LeakyRelu')
    .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor) 
Tensor_tanh = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>('THSTensor_tanh')
    .asFunction();

final Pointer<Void> Function(Pointer<Void>) Tensor_sigmoid = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>('THSTensor_sigmoid')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Void>, Pointer<Int64> result) _allClose = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensorA, Pointer<Void> tensorB, Pointer<Int64> result)>>('AllClose')
    .asFunction();


final Pointer<Utf8> Function(Pointer<Void>, int,int, Pointer<Pointer<Void>> result) _flatten = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Int64 startDim, Int64 endDim, Pointer<Pointer<Void>> result)>>('Flatten')
    .asFunction();

 

    final void Function(Pointer<Void>,Pointer<NativeFunction<AllocatePinnedArrayNative>>, int,int,bool,bool) Tensor_topk = nativeLib
    .lookup<NativeFunction<Void Function(Pointer<Void> tensor,Pointer<NativeFunction<AllocatePinnedArrayNative>>, Int64 k, Int64 dim, Bool largest, Bool sorted)>>('THSTensor_topk')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Void>, Pointer<Pointer<Void>> result)
_expandAs = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Void> other, Pointer<Pointer<Void>> result)>>('ExpandAs')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Void>, Pointer<Pointer<Void>> result)
_eq = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Void> other, Pointer<Pointer<Void>> result)>>('Eq')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, int dim, Pointer<Void> index, Pointer<Pointer<Void>> result)
_indexSelect = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Int64 dim, Pointer<Void> index, Pointer<Pointer<Void>> result)>>('IndexSelect')
    .asFunction();


final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>>,Pointer<Int64>, int)
_view = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void> tensor, Pointer<Pointer<Void>> result,Pointer<Int64> sizes, Int64 sizeLen)>>('View')
    .asFunction();




// 类定义









class Tensor {
  Pointer<Void> _tensorPtr;

  Tensor._internal(this._tensorPtr);
  
  Tensor operator +(dynamic b){
    add_(b);
    return Tensor._internal(_tensorPtr);

  }

   Tensor operator -(dynamic b){
    sub_(b);
    return Tensor._internal(_tensorPtr);

  }

  Tensor operator*(dynamic b){
 mul_(b);
 return Tensor._internal(_tensorPtr);
}

Tensor operator/(dynamic b){
  div_(b);
  return Tensor._internal(_tensorPtr);
}

 Tensor operator[](int index_num){

  return index([index_num],[-1],[-1],[empty([0])]);//In this situation,only indexStarts is useful.See THSTensor_index in src/THSTensor.cpp for more information.
 }




  @override
  String toString() {
    var stringPtr = _print(_tensorPtr);
    final string = stringPtr.cast<Utf8>().toDartString();
    return string;
  }

 Tensor detach() {
    
    
    final resultTensorPtr = Tensor_detach(this._tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);
    
    
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

Tensor add(dynamic b, {double alpha = 1}) {

  if (b is Tensor) {
    final alphaScalar=float64_to_scalar(alpha);

      final resultTensorPtr =
          Tensor_add(_tensorPtr, b._tensorPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      
      return tensor;
    } else if (b is num) {
      if(b is int){
       final alphaScalar=float64_to_scalar(alpha);
        final rightScalar=int32_to_scalar(b);
      final resultTensorPtr =
          Tensor_add_scalar(_tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
       final alphaScalar=float64_to_scalar(alpha);
        final rightScalar=float64_to_scalar(b);
      final resultTensorPtr =
          Tensor_add_scalar(_tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}


Tensor sub( dynamic b, {double alpha = 1}) {

  if (b is Tensor) {
      final alphaScalar=float64_to_scalar(alpha);
        
      final resultTensorPtr =
          Tensor_sub(_tensorPtr,b._tensorPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      final alphaScalar=float64_to_scalar(alpha);
       final rightScalar=int32_to_scalar(b); 
      final resultTensorPtr =
          Tensor_sub_scalar(_tensorPtr,rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
         final alphaScalar=float64_to_scalar(alpha);
       final rightScalar=float64_to_scalar(b); 
      final resultTensorPtr =
          Tensor_sub_scalar(_tensorPtr,rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}


Tensor mul(dynamic b) {


  if (b is Tensor) {
      final resultTensorPtr =
          Tensor_mul(_tensorPtr,b._tensorPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
        final rightScalar=int32_to_scalar(b);
       final resultTensorPtr =
          Tensor_mul_scalar(_tensorPtr,rightScalar.scalarPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        final rightScalar=float64_to_scalar(b);
       final resultTensorPtr =
          Tensor_mul_scalar(_tensorPtr,rightScalar.scalarPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}


Tensor div( dynamic b,{String rounding_mode=""}) {
  
  if (b is Tensor) {
    final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  
final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
      final resultTensorPtr =
          Tensor_div(_tensorPtr,b._tensorPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      final rightScalar=int32_to_scalar(b);

      final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
 final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
       final resultTensorPtr =
          Tensor_div_scalar(_tensorPtr,rightScalar.scalarPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else if(b is double)
      {
       final rightScalar=float64_to_scalar(b);
       final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
       final resultTensorPtr =
          Tensor_div_scalar(_tensorPtr,rightScalar.scalarPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}

  
  void add_(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final alphaScalar=float64_to_scalar(alpha);

      
          Tensor_add_(this._tensorPtr, b._tensorPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

    } else if (b is num) {
      if(b is int){
      final alphaScalar=float64_to_scalar(alpha);
       final rightScalar=int32_to_scalar(b);
      
          Tensor_add_scalar_(this._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      }
      else if(b is double)
      {
        final alphaScalar=float64_to_scalar(alpha);
       final rightScalar=float64_to_scalar(b);
      
          Tensor_add_scalar_(this._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
  }

  void sub_(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final alphaScalar=float64_to_scalar(alpha);

      
          Tensor_sub_(this._tensorPtr, b._tensorPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
    } else if (b is num) {
      if(b is int){
      final alphaScalar=float64_to_scalar(alpha);
       final rightScalar=int32_to_scalar(b);
      
          Tensor_sub_scalar_(this._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      }
      else if(b is double)
      {
        final alphaScalar=float64_to_scalar(alpha);
       final rightScalar=float64_to_scalar(b);
      
          Tensor_sub_scalar_(this._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
  }

 void mul_(dynamic b){
 if (b is Tensor) {
      Tensor_mul_(this._tensorPtr,b._tensorPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      
    } else if (b is num) {
      if(b is int){
      final rightScalar=int32_to_scalar(b);
       
          Tensor_mul_scalar_(this._tensorPtr,rightScalar.scalarPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      
      }
      else if(b is double)
      {
        final rightScalar=float64_to_scalar(b);
      
          Tensor_mul_scalar_(this._tensorPtr,rightScalar.scalarPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}


 void div_(dynamic b){
   String rounding_mode="";
 if (b is Tensor) {
       final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
      
          Tensor_div_(this._tensorPtr,b._tensorPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      
    } else if (b is num) {
      if(b is int){
      final rightScalar=int32_to_scalar(b);

      final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
       
          Tensor_div_scalar_(this._tensorPtr,rightScalar.scalarPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      
      }
      else if(b is double)
      {
        final rightScalar=float64_to_scalar(b);
       final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
       
          Tensor_div_scalar_(this._tensorPtr,rightScalar.scalarPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
      
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}

  
  List<Tensor> convertPointerToTensorList(Pointer<Pointer<Void>> ptr, int count) {
  List<Tensor> tensors = [];
  for (int i = 0; i < count; i++) {
    if(ptr.elementAt(i).value!=nullptr){
    tensors.add(Tensor._internal(ptr.elementAt(i).value));
    }
    else{throw Exception("null Pointer.");}
  }
  return tensors;
}

Pointer<Pointer<Void>> convertListToPointerPointer(List<Tensor> list) {
  final ptr = calloc<Pointer<Void>>(list.length);
  for (int i = 0; i < list.length; i++) {
    ptr[i] = list[i]._tensorPtr;
  }
  return ptr;
}
  Tensor index(List<int> starts,List<int> ends,List<int> steps,List<Tensor> indexTensors)
  {final Pointer<Int64> startPointer = malloc<Int64>(starts.length);
     final Int64List startList = startPointer.asTypedList(starts.length);
    startList.setAll(0, starts);
    final Pointer<Int64> endPointer = malloc<Int64>(ends.length);
     final Int64List endList = endPointer.asTypedList(ends.length);
    endList.setAll(0, ends);
    final Pointer<Int64> stepPointer = malloc<Int64>(steps.length);
     final Int64List stepList = stepPointer.asTypedList(steps.length);
    stepList.setAll(0, steps);
    final indexTensorsPtr=convertListToPointerPointer(indexTensors);
    if(!((starts.length==ends.length)&&(ends.length==steps.length)))
    {throw Exception("wrong input");}

  final resultTensorPtr = Tensor_index(this._tensorPtr,startPointer,endPointer, stepPointer,indexTensorsPtr,starts.length);
    final errorMsg=_get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
  

  return tensor;

  }


Tensor transpose(int dim1, int dim2) {
    
    
    final resultTensorPtr = Tensor_transpose(_tensorPtr, dim1, dim2);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);
    
    
    return tensor;
}





   Tensor permute(List<int> permute_list)
  {final Pointer<Int64> permutePointer = malloc<Int64>(permute_list.length);
     final Int64List permuteList = permutePointer.asTypedList(permute_list.length);
    permuteList.setAll(0, permute_list);

  final resultTensorPtr= Tensor_permute(_tensorPtr,permutePointer,permute_list.length);
  final errorMsg=_get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
  

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

  
List<Tensor> topk(int k, {int dim=-1, bool largest = true, bool sorted = true}) {
  
   
  Tensor_topk(this._tensorPtr, Pointer.fromFunction<AllocatePinnedArrayNative>(allocateMemory),k, dim, largest, sorted);
final errorMsg=_get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }
List<Tensor> tensorList=convertPointerToTensorList(allocator.pointer,2);
return tensorList;
 
}


Tensor expand_as(Tensor other) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _expandAs(this._tensorPtr, other._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}


Tensor eq(Tensor other) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _eq(this._tensorPtr, other._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}

Tensor index_select(int dim, Tensor index) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _indexSelect(this._tensorPtr, dim, index._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}


Tensor view(List<int> size) {
  final sizePtr = calloc<Int64>(size.length);
   final Int64List sizeList = sizePtr.asTypedList(size.length);
    sizeList.setAll(0, size);

 
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _view(this._tensorPtr, resultTensorPtr,sizePtr, size.length);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(sizePtr);
  calloc.free(resultTensorPtr);
  return tensor;
}



}







Tensor eq(Tensor a,Tensor other) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _eq(a._tensorPtr, other._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
}

Tensor index_select(Tensor a,int dim, Tensor index) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _indexSelect(a._tensorPtr, dim, index._tensorPtr, resultTensorPtr);

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);
  return tensor;
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
    
    
    final resultTensorPtr = Tensor_tanh(a._tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);
    
    
    return tensor;
}

Tensor sigmoid(Tensor a) {
  
final resultTensorPtr=Tensor_sigmoid(a._tensorPtr);
final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }
  final tensor = Tensor._internal(resultTensorPtr);
  
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

List<Tensor> topk(Tensor a,int k, {int dim=-1, bool largest = true, bool sorted = true}) {
 return a.topk(k,dim: dim,largest: largest,sorted: sorted);
 
}

Tensor from_blob(List<num> list, List<int> sizes_data,int scalar_type,int dtype,{bool requiresGrad = false,Device ?device_used}) {
   device_used??=device("cpu");
  if (scalar_type==int32) {
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

    final resultTensorPtr =  Tensor_new(
        dataPointer.cast(),Pointer.fromFunction<DeleterNative>(deleteMemory),sizesPointer, sizesList.length, scalar_type, dtype, device_used.device_type,device_used.device_index,requiresGrad);
final errorMsg=_get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  } else if (scalar_type == float32) {
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
    final resultTensorPtr =  Tensor_new(
        dataPointer.cast(),Pointer.fromFunction<DeleterNative>(deleteMemory),sizesPointer, sizesList.length, scalar_type, dtype, device_used.device_type,device_used.device_index,requiresGrad);
final errorMsg=_get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  }
  else if (scalar_type== float64) {
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
    final resultTensorPtr =  Tensor_new(
        dataPointer.cast(),Pointer.fromFunction<DeleterNative>(deleteMemory),sizesPointer, sizesList.length, scalar_type, dtype, device_used.device_type,device_used.device_index,requiresGrad);
final errorMsg=_get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  }
  else {
    throw Exception("wrong type");
  }

  // 使用完毕后释放指针
}





Tensor empty(List<int> size, {bool requiresGrad = false,int dtype=float32,Device ?device_used}) {
  
    device_used??=device("cpu");
  
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = Tensor_empty(int64Pointer, size.length, dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
}

Tensor ones(List<int> size, {bool requiresGrad = false,int dtype=float32,Device ?device_used}) {
  device_used??=device("cpu");
  
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = Tensor_ones(int64Pointer, size.length, dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
}

Tensor full(List<int> size, num values, {int dtype=float32,bool requiresGrad = false,Device ?device_used}) {
 device_used??=device("cpu");
  
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

 if(dtype==float32)
 {
  Scalar scalar=float32_to_scalar(values.toDouble());
  final resultTensorPtr = Tensor_full(int64Pointer, size.length, scalar.scalarPtr,dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
 }
 else if(dtype==float64)
 {
  Scalar scalar=float64_to_scalar(values.toDouble());
  final resultTensorPtr = Tensor_full(int64Pointer, size.length, scalar.scalarPtr,dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
 }
 else if(dtype==int32){
  Scalar scalar=int32_to_scalar(values.toInt());
  final resultTensorPtr = Tensor_full(int64Pointer, size.length, scalar.scalarPtr,dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
 }
 else{throw Exception("wrong type");}
  
}

Tensor eye(int n, int m, {bool requiresGrad = false,int dtype=float32,Device ?device_used}) {
  device_used??=device("cpu");
  
 

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = Tensor_eye(n,m, dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

 

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
}

Tensor arange(num start, num end, num step,{int dtype=float32,bool requiresGrad = false,Device ?device_used}) {
 device_used??=device("cpu");
  
  
 if(dtype==float32)
 {
  Scalar startScalar=float32_to_scalar(start.toDouble());
  Scalar endScalar=float32_to_scalar(start.toDouble());
  Scalar stepScalar=float32_to_scalar(start.toDouble());
  final resultTensorPtr = Tensor_arange(startScalar.scalarPtr,endScalar.scalarPtr,stepScalar.scalarPtr,dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

 

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
 }
 else if(dtype==float64)
 {
  Scalar startScalar=float64_to_scalar(start.toDouble());
  Scalar endScalar=float64_to_scalar(start.toDouble());
  Scalar stepScalar=float64_to_scalar(start.toDouble());
  final resultTensorPtr = Tensor_arange(startScalar.scalarPtr,endScalar.scalarPtr,stepScalar.scalarPtr,dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

 

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
 }
 else if(dtype==int32){
  Scalar startScalar=int32_to_scalar(start.toInt());
  Scalar endScalar=int32_to_scalar(start.toInt());
  Scalar stepScalar=int32_to_scalar(start.toInt());
  final resultTensorPtr = Tensor_arange(startScalar.scalarPtr,endScalar.scalarPtr,stepScalar.scalarPtr,dtype, device_used.device_type,device_used.device_index,requiresGrad);
  final errorMsg =_get_and_reset_last_err();
      

 

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);


  return tensor;
 }
 else{throw Exception("wrong type");}
}

Tensor linspace(double start, double end, int steps,{int dtype=float32,bool requiresGrad = false,Device ?device_used}) {
  device_used??=device("cpu");
  final resultTensorPtr = Tensor_linspace(start, end, steps,dtype,device_used.device_type,device_used.device_index, requiresGrad);
  final errorMsg =_get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor logspace(double start, double end, int steps, double base,
    {int dtype=float32,bool requiresGrad = false,Device ?device_used}) {
      device_used??=device("cpu");
  final resultTensorPtr = Tensor_logspace(start, end, steps,base,dtype,device_used.device_type,device_used.device_index, requiresGrad);
  final errorMsg =_get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
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
  return a.transpose(dim0,dim1);
}

 Tensor permute(Tensor a,List<int> permute_list)
  {final Pointer<Int64> permutePointer = malloc<Int64>(permute_list.length);
     final Int64List permuteList = permutePointer.asTypedList(permute_list.length);
    permuteList.setAll(0, permute_list);

  final resultTensorPtr= Tensor_permute(a._tensorPtr,permutePointer,permute_list.length);
  final errorMsg=_get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
  

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
  return a.add(b,alpha: alpha);
}
else{throw Exception("wrong data type");}
}

Tensor sub(dynamic a, dynamic b, {double alpha = 1}) {
 if(a is num){
  return sub(b,a,alpha: alpha);
}
else if(a is Tensor){
  return a.sub(b,alpha: alpha);
}
else{throw Exception("wrong data type");}
}

Tensor mul(dynamic a, dynamic b) {
if(a is num)
{
  return mul(b,a);
}
else if(a is Tensor){
  return a.mul(b);
}
else{throw Exception("wrong data type.");}
}

Tensor div(dynamic a, dynamic b,{String rounding_mode=""}) {
  if(a is num)
{
  return div(b,a);
}
else if(a is Tensor){
  return a.div(b,rounding_mode: rounding_mode);
}
else{throw Exception("wrong data type.");}
}


Tensor add_(dynamic a, dynamic b, {double alpha = 1}) {
  if(a is num){
  return add_(b,a,alpha: alpha);
}
else if(a is Tensor){
  if (b is Tensor) {
    final alphaScalar=float64_to_scalar(alpha);

      
          Tensor_add_(a._tensorPtr, b._tensorPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      
      return tensor;
    } else if (b is num) {
      if(b is int){
       final alphaScalar=float64_to_scalar(alpha);
        final rightScalar=int32_to_scalar(b);
      
          Tensor_add_scalar_(a._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
      }
      else if(b is double)
      {
       final alphaScalar=float64_to_scalar(alpha);
        final rightScalar=float64_to_scalar(b);
      
          Tensor_add_scalar_(a._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
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
    final alphaScalar=float64_to_scalar(alpha);

      
          Tensor_sub_(a._tensorPtr, b._tensorPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      
      return tensor;
    } else if (b is num) {
      if(b is int){
       final alphaScalar=float64_to_scalar(alpha);
        final rightScalar=int32_to_scalar(b);
      
          Tensor_sub_scalar_(a._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
      }
      else if(b is double)
      {
       final alphaScalar=float64_to_scalar(alpha);
        final rightScalar=float64_to_scalar(b);
      
          Tensor_sub_scalar_(a._tensorPtr, rightScalar.scalarPtr,alphaScalar.scalarPtr );
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
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
      
          Tensor_mul_(a._tensorPtr,b._tensorPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
        final rightScalar=int32_to_scalar(b);
       
          Tensor_mul_scalar_(a._tensorPtr,rightScalar.scalarPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
      }
      else if(b is double)
      {
        final rightScalar=float64_to_scalar(b);
       
          Tensor_mul_scalar_(a._tensorPtr,rightScalar.scalarPtr);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
      }
      else{throw Exception("wrong data type");}
    } else {
      throw Exception("wrong data type.");
    }
}
else{throw Exception("wrong data type.");}
}

Tensor div_(dynamic a, dynamic b,{String rounding_mode=""}) {
  if(a is num)
{
  return div_(b,a);
}
else if(a is Tensor){
  if (b is Tensor) {
    final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  
final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
      
          Tensor_div_(a._tensorPtr,b._tensorPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
    } else if (b is num) {
      if(b is int){
      final rightScalar=int32_to_scalar(b);

      final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
 final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
       
          Tensor_div_scalar_(a._tensorPtr,rightScalar.scalarPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
      }
      else if(b is double)
      {
       final rightScalar=float64_to_scalar(b);
       final units = utf8.encode(rounding_mode);
  // 在本地分配足够的内存来复制这个 Uint8List
  final Pointer<Uint8> result = malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
  // 获取 Uint8List 的指针
  final Uint8List nativeString = result.asTypedList(units.length + 1);
  // 将 Uint8List 复制到分配的内存中
  nativeString.setRange(0, units.length, units);
  // 确保以 null 字节结尾，满足 C 语言对字符串的要求
  nativeString[units.length] = 0;
  // 返回指向已编码字符串的指针
  final rounding_mode_Utf8 = rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
       
          Tensor_div_scalar_(a._tensorPtr,rightScalar.scalarPtr,rounding_mode_Utf8);
      final errorMsg=_get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
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

Tensor_save(a._tensorPtr,path_Utf8);
final errorMsg=_get_and_reset_last_err();
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
  final resultTensorPtr = Tensor_load(path_Utf8);
  final errorMsg=_get_and_reset_last_err();

  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
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
      from_blob(flatList.cast<int>(), sizes,int32,int32);
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
      from_blob(flatList.cast<double>(), sizes,float32,float32);
  return outputTensor;
}

Tensor DoubleTensor(dynamic list) {
   List<num> flatList = [];
  List<int> sizes = [];
  List<bool> isFirstElementAtDepth = [];

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElementAtDepth.length <= depth || isFirstElementAtDepth[depth]) {
        // 如果是首次进入此深度的列表
        if (sizes.length <= depth) {
          sizes.add(element.length); // 添加新尺寸
        } else {
          sizes[depth] = element.length; // 更新此深度的尺寸
        }
        // 扩展或更新首元素标记列表
        if (isFirstElementAtDepth.length <= depth) {
          isFirstElementAtDepth.add(false); // 添加新的深度标记
        } else {
          isFirstElementAtDepth[depth] = false; // 更新此深度的首元素标记
        }
      }

      for (var subElement in element) {
        flatten(subElement, depth + 1);
      }

      // 退出当前列表深度时，重置此深度的首元素标记
      if (isFirstElementAtDepth.length > depth) {
        isFirstElementAtDepth[depth] = true;
      }
    } else if (element is num) {
      flatList.add(element);
    }
  }

  flatten(list, 0);

 

  Tensor outputTensor =
      from_blob(flatList.cast<double>(), sizes,float64,float64);
  return outputTensor;
}

Scalar int32_to_scalar(int value){
final result=_int32_to_scalar(value);
final errorMsg=_get_and_reset_last_err();
if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }
  final scalar=Scalar(result);
  return scalar;

}

Scalar float32_to_scalar(double value){
final result=_float32_to_scalar(value);
final errorMsg=_get_and_reset_last_err();
if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }
  final scalar=Scalar(result);
  return scalar;

}

Scalar float64_to_scalar(double value){
final result=_float64_to_scalar(value);
final errorMsg=_get_and_reset_last_err();
if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    
    throw Exception(errorString);
  }
  final scalar=Scalar(result);
  return scalar;

}