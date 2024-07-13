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

final Pointer<Utf8> Function() _get_and_reset_last_err = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function()>>(
        'THSTorch_get_and_reset_last_err')
    .asFunction();

final Pointer<Void> Function(int value) _int32_to_scalar = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Int value)>>(
        'THSTorch_int32_to_scalar')
    .asFunction();

final Pointer<Void> Function(double value) _float32_to_scalar = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Float value)>>(
        'THSTorch_float32_to_scalar')
    .asFunction();
final Pointer<Void> Function(double value) _float64_to_scalar = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Double value)>>(
        'THSTorch_float64_to_scalar')
    .asFunction();
final Pointer<Void> Function(Pointer<Void> tensor) Tensor_clone = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>(
        'THSTensor_clone')
    .asFunction();

final Pointer<Void> Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_add = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_add')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_add_scalar = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_add_scalar')
        .asFunction();

final void Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_add_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_add_')
        .asFunction();

final void Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_add_scalar_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_add_scalar_')
        .asFunction();

final void Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_sub_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_sub_')
        .asFunction();

final void Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_sub_scalar_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_sub_scalar_')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_sub = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_sub')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Void> alpha)
    Tensor_sub_scalar = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Void> alpha)>>('THSTensor_sub_scalar')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right)
    Tensor_mul = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left, Pointer<Void> right)>>('THSTensor_mul')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right)
    Tensor_mul_scalar = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left,
                    Pointer<Void> right)>>('THSTensor_mul_scalar')
        .asFunction();

final void Function(Pointer<Void> left, Pointer<Void> right) Tensor_mul_ =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> left, Pointer<Void> right)>>('THSTensor_mul_')
        .asFunction();

final void Function(Pointer<Void> left, Pointer<Void> right)
    Tensor_mul_scalar_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left,
                    Pointer<Void> right)>>('THSTensor_mul_scalar_')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Utf8> rounding_mode)
    Tensor_div = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Utf8> rounding_mode)>>('THSTensor_div')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Utf8> rounding_mode)
    Tensor_div_scalar = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Utf8> rounding_mode)>>('THSTensor_div_scalar')
        .asFunction();

final void Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Utf8> rounding_mode)
    Tensor_div_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Utf8> rounding_mode)>>('THSTensor_div_')
        .asFunction();

final void Function(
        Pointer<Void> left, Pointer<Void> right, Pointer<Utf8> rounding_mode)
    Tensor_div_scalar_ = nativeLib
        .lookup<
            NativeFunction<
                Void Function(Pointer<Void> left, Pointer<Void> right,
                    Pointer<Utf8> rounding_mode)>>('THSTensor_div_scalar_')
        .asFunction();
final Pointer<Void> Function(Pointer<Int64> sizes, int length, int scalar_type,
        int device_type, int device_index, bool requires_grad) Tensor_empty =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Int64> sizes,
                    Int32 length,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_empty')
        .asFunction();

final Pointer<Void> Function(Pointer<Int64> sizes, int length, int scalar_type,
        int device_type, int device_index, bool requires_grad) Tensor_ones =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Int64> sizes,
                    Int32 length,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_ones')
        .asFunction();
final Pointer<Void> Function(int n, int m, int scalar_type, int device_type,
        int device_index, bool requires_grad) Tensor_eye =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Int32 n,
                    Int32 m,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_eye')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Int64> sizes,
        int length,
        Pointer<Void> scalar,
        int scalar_type,
        int device_type,
        int device_index,
        bool requires_grad) Tensor_full =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Int64> sizes,
                    Int32 length,
                    Pointer<Void> scalar,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_full')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void>) _print = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void>)>>(
        'Tensor_Print')
    .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor) Tensor_detach = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>(
        'THSTensor_detach')
    .asFunction();

final Pointer<Void> Function(Pointer<Void>, Pointer<Int64>, Pointer<Int64>,
        Pointer<Int64>, Pointer<Pointer<Void>>, int) Tensor_index =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void>,
                    Pointer<Int64>,
                    Pointer<Int64>,
                    Pointer<Int64>,
                    Pointer<Pointer<Void>>,
                    Int64)>>('THSTensor_index')
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

final int Function(Pointer<Void>) Tensor_type = nativeLib
    .lookup<NativeFunction<Int8 Function(Pointer<Void>)>>('THSTensor_type')
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
        bool requires_grad) Tensor_new =
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
                    Bool requires_grad)>>('THSTensor_new')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void> a, int dtype, Pointer<Int> result)
    _toList_Int = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> a, Int8 dtype,
                    Pointer<Int> result)>>('Tensor_ToArray_Int')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void> a, int dtype, Pointer<Int64> result)
    _toList_Int64 = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> a, Int8 dtype,
                    Pointer<Int64> result)>>('Tensor_ToArray_Int64')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void> a, int dtype, Pointer<Float> result)
    _toList_Float = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> a, Int8 dtype,
                    Pointer<Float> result)>>('Tensor_ToArray_Float')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void> a, int dtype, Pointer<Double> result)
    _toList_Float64 = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> a, Int8 dtype,
                    Pointer<Double> result)>>('Tensor_ToArray_Float64')
        .asFunction();

final Pointer<Void> Function(
        Pointer<Void> start,
        Pointer<Void> end,
        Pointer<Void> step,
        int scalar_type,
        int device_type,
        int device_index,
        bool requires_grad) Tensor_arange =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> start,
                    Pointer<Void> end,
                    Pointer<Void> step,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_arange')
        .asFunction();

final Pointer<Void> Function(double start, double end, int steps,
        int scalar_type, int device_type, int device_index, bool requires_grad)
    Tensor_linspace = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Double start,
                    Double end,
                    Int64 steps,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_linspace')
        .asFunction();

final Pointer<Void> Function(double start, double end, int steps, double base,
        int scalar_type, int device_type, int device_index, bool requires_grad)
    Tensor_logspace = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Double start,
                    Double end,
                    Int64 steps,
                    Double base,
                    Int8 scalar_type,
                    Int32 device_type,
                    Int32 device_index,
                    Bool requires_grad)>>('THSTensor_logspace')
        .asFunction();

final int Function(Pointer<Void> left, Pointer<Void> right) Tensor_equal =
    nativeLib
        .lookup<
            NativeFunction<
                Int32 Function(Pointer<Void> left,
                    Pointer<Void> right)>>('THSTensor_equal')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor, bool has_type, int dtype)
    Tensor_sum = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor, Bool has_type,
                    Int8 dtype)>>('THSTensor_sum')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right)
    Tensor_mm = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left, Pointer<Void> right)>>('THSTensor_mm')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor, int dim1, int dim2)
    Tensor_transpose = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor, Int64 dim1,
                    Int64 dim2)>>('THSTensor_transpose')
        .asFunction();

final Pointer<Void> Function(Pointer<Void>, Pointer<Int64>, int dim_size)
    Tensor_permute = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                  Pointer<Void> tensor,
                  Pointer<Int64>,
                  Int64 dim_size,
                )>>('THSTensor_permute')
        .asFunction();

final void Function(Pointer<Void> a, Pointer<Utf8> path) Tensor_save = nativeLib
    .lookup<NativeFunction<Void Function(Pointer<Void> a, Pointer<Utf8>)>>(
        'THSTensor_save')
    .asFunction();
final Pointer<Void> Function(
  Pointer<Utf8> path,
) Tensor_load = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Utf8> path)>>(
        'THSTensor_load')
    .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor) Tensor_relu = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>(
        'THSTensor_relu')
    .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor, Pointer<Void> negative_slope)
    Tensor_leaky_relu = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor,
                    Pointer<Void> negative_slope)>>('THSTensor_leaky_relu')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> tensor) Tensor_tanh = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>(
        'THSTensor_tanh')
    .asFunction();

final Pointer<Void> Function(Pointer<Void>) Tensor_sigmoid = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void> tensor)>>(
        'THSTensor_sigmoid')
    .asFunction();

final int Function(Pointer<Void> left, Pointer<Void> right, double rtol,
        double atol, bool equal_nan) Tensor_allclose =
    nativeLib
        .lookup<
            NativeFunction<
                Int32 Function(
                    Pointer<Void> left,
                    Pointer<Void> right,
                    Double rtol,
                    Double atol,
                    Bool equal_nan)>>('THSTensor_allclose')
        .asFunction();

final Pointer<Void> Function(Pointer<Void>, int, int) Tensor_flatten = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Void> Function(Pointer<Void> tensor, Int64 startDim,
                Int64 endDim)>>('THSTensor_flatten')
    .asFunction();

final void Function(
        Pointer<Void>,
        Pointer<NativeFunction<AllocatePinnedArrayNative>>,
        int,
        int,
        bool,
        bool) Tensor_topk =
    nativeLib
        .lookup<
            NativeFunction<
                Void Function(
                    Pointer<Void> tensor,
                    Pointer<NativeFunction<AllocatePinnedArrayNative>>,
                    Int64 k,
                    Int64 dim,
                    Bool largest,
                    Bool sorted)>>('THSTensor_topk')
        .asFunction();

final Pointer<Void> Function(Pointer<Void>, Pointer<Void>) Tensor_expand_as =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor,
                    Pointer<Void> other)>>('THSTensor_expand_as')
        .asFunction();

final Pointer<Void> Function(Pointer<Void> left, Pointer<Void> right)
    Tensor_eq = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(
                    Pointer<Void> left, Pointer<Void> right)>>('THSTensor_eq')
        .asFunction();

final Pointer<Void> Function(Pointer<Void>, int dim, Pointer<Void> index)
    Tensor_index_select = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor, Int64 dim,
                    Pointer<Void> index)>>('THSTensor_index_select')
        .asFunction();

final Pointer<Void> Function(Pointer<Void>, Pointer<Int64>, int) Tensor_view =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Void> Function(Pointer<Void> tensor,
                    Pointer<Int64> sizes, Int64 sizeLen)>>('THSTensor_view')
        .asFunction();

final Pointer<Void> Function(Pointer<Void>, int) Tensor_unsqueeze = nativeLib
    .lookup<
        NativeFunction<
            Pointer<Void> Function(
                Pointer<Void> tensor, Int64 dim)>>('THSTensor_unsqueeze')
    .asFunction();

// 类定义

class Tensor {
  Pointer<Void> _tensorPtr;

  Tensor._internal(this._tensorPtr);
  Tensor(Pointer<Void> tensorPointer) : _tensorPtr = tensorPointer;
  Tensor operator +(dynamic b) {
    add_(b);
    return Tensor._internal(_tensorPtr);
  }

  Tensor operator -(dynamic b) {
    sub_(b);
    return Tensor._internal(_tensorPtr);
  }

  Tensor operator *(dynamic b) {
    mul_(b);
    return Tensor._internal(_tensorPtr);
  }

  Tensor operator /(dynamic b) {
    div_(b);
    return Tensor._internal(_tensorPtr);
  }

  Tensor operator [](int index_num) {
    return index([
      index_num
    ], [
      -1
    ], [
      -1
    ], [
      empty([0])
    ]); //In this situation,only indexStarts is useful.See THSTensor_index in src/THSTensor.cpp for more information.
  }

  Pointer<Void> get tensorPtr => _tensorPtr;
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
    final dtype = Tensor_type(_tensorPtr);
    final errorMsg = _get_and_reset_last_err();

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

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
      final alphaScalar = float64_to_scalar(alpha);

      final resultTensorPtr =
          Tensor_add(_tensorPtr, b._tensorPtr, alphaScalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);

      return tensor;
    } else if (b is num) {
      if (b is int) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = int32_to_scalar(b);
        final resultTensorPtr = Tensor_add_scalar(
            _tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else if (b is double) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = float64_to_scalar(b);
        final resultTensorPtr = Tensor_add_scalar(
            _tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  Tensor sub(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final alphaScalar = float64_to_scalar(alpha);

      final resultTensorPtr =
          Tensor_sub(_tensorPtr, b._tensorPtr, alphaScalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if (b is int) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = int32_to_scalar(b);
        final resultTensorPtr = Tensor_sub_scalar(
            _tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else if (b is double) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = float64_to_scalar(b);
        final resultTensorPtr = Tensor_sub_scalar(
            _tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  Tensor mul(dynamic b) {
    if (b is Tensor) {
      final resultTensorPtr = Tensor_mul(_tensorPtr, b._tensorPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if (b is int) {
        final rightScalar = int32_to_scalar(b);
        final resultTensorPtr =
            Tensor_mul_scalar(_tensorPtr, rightScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else if (b is double) {
        final rightScalar = float64_to_scalar(b);
        final resultTensorPtr =
            Tensor_mul_scalar(_tensorPtr, rightScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  Tensor div(dynamic b, {String rounding_mode = ""}) {
    if (b is Tensor) {
      final units = utf8.encode(rounding_mode);
      // 在本地分配足够的内存来复制这个 Uint8List
      final Pointer<Uint8> result =
          malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
      // 获取 Uint8List 的指针
      final Uint8List nativeString = result.asTypedList(units.length + 1);
      // 将 Uint8List 复制到分配的内存中
      nativeString.setRange(0, units.length, units);
      // 确保以 null 字节结尾，满足 C 语言对字符串的要求
      nativeString[units.length] = 0;
      // 返回指向已编码字符串的指针

      final rounding_mode_Utf8 =
          rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
      final resultTensorPtr =
          Tensor_div(_tensorPtr, b._tensorPtr, rounding_mode_Utf8);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);
      return tensor;
    } else if (b is num) {
      if (b is int) {
        final rightScalar = int32_to_scalar(b);

        final units = utf8.encode(rounding_mode);
        // 在本地分配足够的内存来复制这个 Uint8List
        final Pointer<Uint8> result =
            malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
        // 获取 Uint8List 的指针
        final Uint8List nativeString = result.asTypedList(units.length + 1);
        // 将 Uint8List 复制到分配的内存中
        nativeString.setRange(0, units.length, units);
        // 确保以 null 字节结尾，满足 C 语言对字符串的要求
        nativeString[units.length] = 0;
        // 返回指向已编码字符串的指针
        final rounding_mode_Utf8 =
            rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
        final resultTensorPtr = Tensor_div_scalar(
            _tensorPtr, rightScalar.scalarPtr, rounding_mode_Utf8);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else if (b is double) {
        final rightScalar = float64_to_scalar(b);
        final units = utf8.encode(rounding_mode);
        // 在本地分配足够的内存来复制这个 Uint8List
        final Pointer<Uint8> result =
            malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
        // 获取 Uint8List 的指针
        final Uint8List nativeString = result.asTypedList(units.length + 1);
        // 将 Uint8List 复制到分配的内存中
        nativeString.setRange(0, units.length, units);
        // 确保以 null 字节结尾，满足 C 语言对字符串的要求
        nativeString[units.length] = 0;
        // 返回指向已编码字符串的指针
        final rounding_mode_Utf8 =
            rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();
        final resultTensorPtr = Tensor_div_scalar(
            _tensorPtr, rightScalar.scalarPtr, rounding_mode_Utf8);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(resultTensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  void add_(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final alphaScalar = float64_to_scalar(alpha);

      Tensor_add_(this._tensorPtr, b._tensorPtr, alphaScalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
    } else if (b is num) {
      if (b is int) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = int32_to_scalar(b);

        Tensor_add_scalar_(
            this._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else if (b is double) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = float64_to_scalar(b);

        Tensor_add_scalar_(
            this._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  void sub_(dynamic b, {double alpha = 1}) {
    if (b is Tensor) {
      final alphaScalar = float64_to_scalar(alpha);

      Tensor_sub_(this._tensorPtr, b._tensorPtr, alphaScalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
    } else if (b is num) {
      if (b is int) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = int32_to_scalar(b);

        Tensor_sub_scalar_(
            this._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else if (b is double) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = float64_to_scalar(b);

        Tensor_sub_scalar_(
            this._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  void mul_(dynamic b) {
    if (b is Tensor) {
      Tensor_mul_(this._tensorPtr, b._tensorPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
    } else if (b is num) {
      if (b is int) {
        final rightScalar = int32_to_scalar(b);

        Tensor_mul_scalar_(this._tensorPtr, rightScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else if (b is double) {
        final rightScalar = float64_to_scalar(b);

        Tensor_mul_scalar_(this._tensorPtr, rightScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  void div_(dynamic b) {
    String rounding_mode = "";
    if (b is Tensor) {
      final units = utf8.encode(rounding_mode);
      // 在本地分配足够的内存来复制这个 Uint8List
      final Pointer<Uint8> result =
          malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
      // 获取 Uint8List 的指针
      final Uint8List nativeString = result.asTypedList(units.length + 1);
      // 将 Uint8List 复制到分配的内存中
      nativeString.setRange(0, units.length, units);
      // 确保以 null 字节结尾，满足 C 语言对字符串的要求
      nativeString[units.length] = 0;
      // 返回指向已编码字符串的指针
      final rounding_mode_Utf8 =
          rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();

      Tensor_div_(this._tensorPtr, b._tensorPtr, rounding_mode_Utf8);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
    } else if (b is num) {
      if (b is int) {
        final rightScalar = int32_to_scalar(b);

        final units = utf8.encode(rounding_mode);
        // 在本地分配足够的内存来复制这个 Uint8List
        final Pointer<Uint8> result =
            malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
        // 获取 Uint8List 的指针
        final Uint8List nativeString = result.asTypedList(units.length + 1);
        // 将 Uint8List 复制到分配的内存中
        nativeString.setRange(0, units.length, units);
        // 确保以 null 字节结尾，满足 C 语言对字符串的要求
        nativeString[units.length] = 0;
        // 返回指向已编码字符串的指针
        final rounding_mode_Utf8 =
            rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();

        Tensor_div_scalar_(
            this._tensorPtr, rightScalar.scalarPtr, rounding_mode_Utf8);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else if (b is double) {
        final rightScalar = float64_to_scalar(b);
        final units = utf8.encode(rounding_mode);
        // 在本地分配足够的内存来复制这个 Uint8List
        final Pointer<Uint8> result =
            malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
        // 获取 Uint8List 的指针
        final Uint8List nativeString = result.asTypedList(units.length + 1);
        // 将 Uint8List 复制到分配的内存中
        nativeString.setRange(0, units.length, units);
        // 确保以 null 字节结尾，满足 C 语言对字符串的要求
        nativeString[units.length] = 0;
        // 返回指向已编码字符串的指针
        final rounding_mode_Utf8 =
            rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();

        Tensor_div_scalar_(
            this._tensorPtr, rightScalar.scalarPtr, rounding_mode_Utf8);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  }

  List<Tensor> convertPointerToTensorList(
      Pointer<Pointer<Void>> ptr, int count) {
    List<Tensor> tensors = [];
    for (int i = 0; i < count; i++) {
      if (ptr.elementAt(i).value != nullptr) {
        tensors.add(Tensor._internal(ptr.elementAt(i).value));
      } else {
        throw Exception("null Pointer.");
      }
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

  Tensor index(List<int> starts, List<int> ends, List<int> steps,
      List<Tensor> indexTensors) {
    final Pointer<Int64> startPointer = malloc<Int64>(starts.length);
    final Int64List startList = startPointer.asTypedList(starts.length);
    startList.setAll(0, starts);
    final Pointer<Int64> endPointer = malloc<Int64>(ends.length);
    final Int64List endList = endPointer.asTypedList(ends.length);
    endList.setAll(0, ends);
    final Pointer<Int64> stepPointer = malloc<Int64>(steps.length);
    final Int64List stepList = stepPointer.asTypedList(steps.length);
    stepList.setAll(0, steps);
    final indexTensorsPtr = convertListToPointerPointer(indexTensors);
    if (!((starts.length == ends.length) && (ends.length == steps.length))) {
      throw Exception("wrong input");
    }

    final resultTensorPtr = Tensor_index(this._tensorPtr, startPointer,
        endPointer, stepPointer, indexTensorsPtr, starts.length);
    final errorMsg = _get_and_reset_last_err();
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

  Tensor permute(List<int> permute_list) {
    final Pointer<Int64> permutePointer = malloc<Int64>(permute_list.length);
    final Int64List permuteList =
        permutePointer.asTypedList(permute_list.length);
    permuteList.setAll(0, permute_list);

    final resultTensorPtr =
        Tensor_permute(_tensorPtr, permutePointer, permute_list.length);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  List<dynamic> toList() {
    var Dtype = this.dtype();
    var tensorLength = this.shape().reduce((value, element) => value * element);
    List<int> tensorShape = this.shape();
    if (Dtype == int32) {
      // 获取该数组的指针

      // 创建 sizes 数组的指针

      final resultListPtr = calloc<Int>(tensorLength);

      final errorMsg = _toList_Int(this._tensorPtr, Dtype, resultListPtr);

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
          var sublist =
              buildList(dimension + 1, offset + i * strides[dimension]);
          if (dimension == 0) {
            result.addAll(sublist);
          } else {
            result.add(sublist);
          }
        }
        return result;
      }

      return buildList(0, 0);
    } else if (Dtype == int64) {
      // 获取该数组的指针

      // 创建 sizes 数组的指针

      final resultListPtr = calloc<Int64>(tensorLength);

      final errorMsg = _toList_Int64(this._tensorPtr, Dtype, resultListPtr);

      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
      final Pointer<Int64> dataPointer = resultListPtr.cast<Int64>();

      final Int64List flatList = dataPointer.asTypedList(tensorLength);

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
          var sublist =
              buildList(dimension + 1, offset + i * strides[dimension]);
          if (dimension == 0) {
            result.addAll(sublist);
          } else {
            result.add(sublist);
          }
        }
        return result;
      }

      return buildList(0, 0);
    } else if (Dtype == float32) {
      // 获取该数组的指针

      // 创建 sizes 数组的指针

      final resultListPtr = calloc<Float>(tensorLength);

      final errorMsg = _toList_Float(this._tensorPtr, Dtype, resultListPtr);

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
          var sublist =
              buildList(dimension + 1, offset + i * strides[dimension]);
          if (dimension == 0) {
            result.addAll(sublist);
          } else {
            result.add(sublist);
          }
        }
        return result;
      }

      return buildList(0, 0);
    } else if (Dtype == float64) {
      // 获取该数组的指针

      // 创建 sizes 数组的指针

      final resultListPtr = calloc<Double>(tensorLength);

      final errorMsg = _toList_Float64(this._tensorPtr, Dtype, resultListPtr);

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
          var sublist =
              buildList(dimension + 1, offset + i * strides[dimension]);
          if (dimension == 0) {
            result.addAll(sublist);
          } else {
            result.add(sublist);
          }
        }
        return result;
      }

      return buildList(0, 0);
    } else {
      throw Exception("wrong type");
    }

    // 使用完毕后释放指针
  }

  List<Tensor> topk(int k,
      {int dim = -1, bool largest = true, bool sorted = true}) {
    Tensor_topk(
        this._tensorPtr,
        Pointer.fromFunction<AllocatePinnedArrayNative>(allocateMemory),
        k,
        dim,
        largest,
        sorted);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }
    List<Tensor> tensorList = convertPointerToTensorList(allocator.pointer, 2);
    return tensorList;
  }

  Tensor expand_as(Tensor other) {
    final resultTensorPtr = Tensor_expand_as(this._tensorPtr, other._tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor eq(Tensor right) {
    final resultTensorPtr = Tensor_eq(_tensorPtr, right._tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor unsqueeze(int dim) {
    final resultTensorPtr = Tensor_unsqueeze(_tensorPtr, dim);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor index_select(int dim, Tensor index) {
    final resultTensorPtr =
        Tensor_index_select(this._tensorPtr, dim, index._tensorPtr);

    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor view(List<int> size) {
    final sizePtr = calloc<Int64>(size.length);
    final Int64List sizeList = sizePtr.asTypedList(size.length);
    sizeList.setAll(0, size);

    final resultTensorPtr = Tensor_view(this._tensorPtr, sizePtr, size.length);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);
    calloc.free(sizePtr);

    return tensor;
  }

  Tensor relu() {
    final resultTensorPtr = Tensor_relu(_tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor leaky_relu(dynamic negative_slope) {
    if (negative_slope is int) {
      final negative_slope_scalar = int32_to_scalar(negative_slope);
      final resultTensorPtr =
          Tensor_leaky_relu(_tensorPtr, negative_slope_scalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);

      return tensor;
    } else if (negative_slope is double) {
      final negative_slope_scalar = float64_to_scalar(negative_slope);
      final resultTensorPtr =
          Tensor_leaky_relu(_tensorPtr, negative_slope_scalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();
        throw Exception(errorString);
      }

      final tensor = Tensor._internal(resultTensorPtr);

      return tensor;
    } else {
      throw Exception("wrong data type.");
    }
  }

  Tensor tanh() {
    final resultTensorPtr = Tensor_tanh(_tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor sigmoid() {
    final resultTensorPtr = Tensor_sigmoid(_tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor flatten(int startDim, int endDim) {
    final resultTensorPtr = Tensor_flatten(_tensorPtr, startDim, endDim);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  bool equal(Tensor right) {
    final resultPtr = Tensor_equal(_tensorPtr, right._tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final value = resultPtr > 0 ? true : false;

    return value;
  }

  Tensor sum({bool has_type = false, int Dtype = float32}) {
    if (has_type == false) {
      Dtype = dtype();
    }

    final resultTensorPtr = Tensor_sum(_tensorPtr, has_type, Dtype);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  bool allclose(Tensor right,
      {double rtol = 1e-08, double atol = 1e-05, bool equal_nan = false}) {
    final result =
        Tensor_allclose(_tensorPtr, right._tensorPtr, rtol, atol, equal_nan);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final value = result > 0 ? true : false;
    return value;
  }

  Tensor mm(Tensor right) {
    final resultTensorPtr = Tensor_mm(_tensorPtr, right._tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }

  Tensor clone() {
    final resultTensorPtr = Tensor_clone(_tensorPtr);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  }
}

Tensor unsqueeze(Tensor tensor, int dim) {
  return tensor.unsqueeze(dim);
}

Tensor clone(Tensor tensor) {
  return tensor.clone();
}

Tensor relu(Tensor a) {
  return a.relu();
}

Tensor leaky_relu(Tensor a, dynamic negative_slope) {
  return a.leaky_relu(negative_slope);
}

Tensor tanh(Tensor a) {
  return a.tanh();
}

Tensor sigmoid(Tensor a) {
  return a.sigmoid();
}

Tensor eq(Tensor a, Tensor other) {
  return a.eq(other);
}

Tensor index_select(Tensor a, int dim, Tensor index) {
  return a.index_select(dim, index);
}

bool allclose(Tensor left, Tensor right,
    {double rtol = 1e-08, double atol = 1e-05, bool equal_nan = false}) {
  return left.allclose(right, atol: atol, rtol: rtol, equal_nan: equal_nan);
}

Tensor flatten(Tensor a, int startDim, int endDim) {
  return a.flatten(startDim, endDim);
}

List<Tensor> topk(Tensor a, int k,
    {int dim = -1, bool largest = true, bool sorted = true}) {
  return a.topk(k, dim: dim, largest: largest, sorted: sorted);
}

Tensor from_blob(
    List<num> list, List<int> sizes_data, int scalar_type, int dtype,
    {bool requiresGrad = false, Device? device_used}) {
  device_used ??= device("cpu");
  if (scalar_type == int32) {
    Int32List intList;
    if(list is Int32List)
    {
        intList=list;
    }
    else{
    intList = Int32List.fromList(list.cast<int>());
    }
    // 获取该数组的指针
    final Pointer<Int32> dataPointer = malloc<Int32>(intList.length);
    dataPointer
        .asTypedList(intList.length)
        .setRange(0, intList.length, intList);

    // 创建 sizes 数组的指针
    final Pointer<Int64> sizesPointer = malloc<Int64>(sizes_data.length);
    final Int64List sizesList = sizesPointer.asTypedList(sizes_data.length);
    sizesList.setAll(0, sizes_data);

    final resultTensorPtr = Tensor_new(
        dataPointer.cast(),
        Pointer.fromFunction<DeleterNative>(deleteMemory),
        sizesPointer,
        sizesList.length,
        scalar_type,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  } 
  else if (scalar_type == int64) {
    Int64List intList;
    if(list is Int64List)
    {
      intList=list;
    }
    else{
    intList = Int64List.fromList(list.cast<int>());
    }
    // 获取该数组的指针
    final Pointer<Int64> dataPointer = malloc<Int64>(intList.length);
    dataPointer
        .asTypedList(intList.length)
        .setRange(0, intList.length, intList);

    // 创建 sizes 数组的指针
    final Pointer<Int64> sizesPointer = malloc<Int64>(sizes_data.length);
    final Int64List sizesList = sizesPointer.asTypedList(sizes_data.length);
    sizesList.setAll(0, sizes_data);

    // 调用 FFI 函数

    // 调用 FFI 函数
    final resultTensorPtr = Tensor_new(
        dataPointer.cast(),
        Pointer.fromFunction<DeleterNative>(deleteMemory),
        sizesPointer,
        sizesList.length,
        scalar_type,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  }
  
  else if (scalar_type == float32) {
    Float32List floatList;
    if(list is Float32List)
    {
      floatList=list;
    }
    else{
     floatList = Float32List.fromList(list.cast<double>());
    }
    // 获取该数组的指针
    final Pointer<Float> dataPointer = malloc<Float>(floatList.length);
    dataPointer
        .asTypedList(floatList.length)
        .setRange(0, floatList.length, floatList);

    // 创建 sizes 数组的指针
    final Pointer<Int64> sizesPointer = malloc<Int64>(sizes_data.length);
    final Int64List sizesList = sizesPointer.asTypedList(sizes_data.length);
    sizesList.setAll(0, sizes_data);
    print(sizesList);
    // 调用 FFI 函数

    // 调用 FFI 函数
    final resultTensorPtr = Tensor_new(
        dataPointer.cast(),
        Pointer.fromFunction<DeleterNative>(deleteMemory),
        sizesPointer,
        sizesList.length,
        scalar_type,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  } 
  else if (scalar_type == float32) {
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
    final resultTensorPtr = Tensor_new(
        dataPointer.cast(),
        Pointer.fromFunction<DeleterNative>(deleteMemory),
        sizesPointer,
        sizesList.length,
        scalar_type,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  }
  else if (scalar_type == float64) {
    Float64List floatList;
    if(list is Float64List)
    {
      floatList=list;
    }
    else{
    floatList = Float64List.fromList(list.cast<double>());
    }
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
    final resultTensorPtr = Tensor_new(
        dataPointer.cast(),
        Pointer.fromFunction<DeleterNative>(deleteMemory),
        sizesPointer,
        sizesList.length,
        scalar_type,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr);
    return tensor;
  } else {
    throw Exception("wrong type");
  }

  // 使用完毕后释放指针
}

Tensor empty(List<int> size,
    {bool requiresGrad = false, int dtype = float32, Device? device_used}) {
  device_used ??= device("cpu");

  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = Tensor_empty(int64Pointer, size.length, dtype,
      device_used.device_type, device_used.device_index, requiresGrad);
  final errorMsg = _get_and_reset_last_err();

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

Tensor ones(List<int> size,
    {bool requiresGrad = false, int dtype = float32, Device? device_used}) {
  device_used ??= device("cpu");

  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = Tensor_ones(int64Pointer, size.length, dtype,
      device_used.device_type, device_used.device_index, requiresGrad);
  final errorMsg = _get_and_reset_last_err();

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

Tensor full(List<int> size, num values,
    {int dtype = float32, bool requiresGrad = false, Device? device_used}) {
  device_used ??= device("cpu");

  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  if (dtype == float32) {
    Scalar scalar = float32_to_scalar(values.toDouble());
    final resultTensorPtr = Tensor_full(
        int64Pointer,
        size.length,
        scalar.scalarPtr,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();

    // 释放原生数组内存
    calloc.free(int64Pointer);

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else if (dtype == float64) {
    Scalar scalar = float64_to_scalar(values.toDouble());
    final resultTensorPtr = Tensor_full(
        int64Pointer,
        size.length,
        scalar.scalarPtr,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();

    // 释放原生数组内存
    calloc.free(int64Pointer);

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else if (dtype == int32) {
    Scalar scalar = int32_to_scalar(values.toInt());
    final resultTensorPtr = Tensor_full(
        int64Pointer,
        size.length,
        scalar.scalarPtr,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();

    // 释放原生数组内存
    calloc.free(int64Pointer);

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else {
    throw Exception("wrong type");
  }
}

Tensor eye(int n, int m,
    {bool requiresGrad = false, int dtype = float32, Device? device_used}) {
  device_used ??= device("cpu");

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = Tensor_eye(n, m, dtype, device_used.device_type,
      device_used.device_index, requiresGrad);
  final errorMsg = _get_and_reset_last_err();

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);

  return tensor;
}

Tensor arange(num start, num end, num step,
    {int dtype = float32, bool requiresGrad = false, Device? device_used}) {
  device_used ??= device("cpu");

  if (dtype == float32) {
    Scalar startScalar = float32_to_scalar(start.toDouble());
    Scalar endScalar = float32_to_scalar(start.toDouble());
    Scalar stepScalar = float32_to_scalar(start.toDouble());
    final resultTensorPtr = Tensor_arange(
        startScalar.scalarPtr,
        endScalar.scalarPtr,
        stepScalar.scalarPtr,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else if (dtype == float64) {
    Scalar startScalar = float64_to_scalar(start.toDouble());
    Scalar endScalar = float64_to_scalar(start.toDouble());
    Scalar stepScalar = float64_to_scalar(start.toDouble());
    final resultTensorPtr = Tensor_arange(
        startScalar.scalarPtr,
        endScalar.scalarPtr,
        stepScalar.scalarPtr,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else if (dtype == int32) {
    Scalar startScalar = int32_to_scalar(start.toInt());
    Scalar endScalar = int32_to_scalar(start.toInt());
    Scalar stepScalar = int32_to_scalar(start.toInt());
    final resultTensorPtr = Tensor_arange(
        startScalar.scalarPtr,
        endScalar.scalarPtr,
        stepScalar.scalarPtr,
        dtype,
        device_used.device_type,
        device_used.device_index,
        requiresGrad);
    final errorMsg = _get_and_reset_last_err();

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else {
    throw Exception("wrong type");
  }
}

Tensor linspace(double start, double end, int steps,
    {int dtype = float32, bool requiresGrad = false, Device? device_used}) {
  device_used ??= device("cpu");
  final resultTensorPtr = Tensor_linspace(start, end, steps, dtype,
      device_used.device_type, device_used.device_index, requiresGrad);
  final errorMsg = _get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor logspace(double start, double end, int steps, double base,
    {int dtype = float32, bool requiresGrad = false, Device? device_used}) {
  device_used ??= device("cpu");
  final resultTensorPtr = Tensor_logspace(start, end, steps, base, dtype,
      device_used.device_type, device_used.device_index, requiresGrad);
  final errorMsg = _get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);
  calloc.free(resultTensorPtr);

  return tensor;
}

bool equal(Tensor left, Tensor right) {
  return left.equal(right);
}

Tensor mm(Tensor left, Tensor right) {
  return left.mm(right);
}

Tensor transpose(Tensor a, int dim0, int dim1) {
  return a.transpose(dim0, dim1);
}

Tensor permute(Tensor a, List<int> permute_list) {
  final Pointer<Int64> permutePointer = malloc<Int64>(permute_list.length);
  final Int64List permuteList = permutePointer.asTypedList(permute_list.length);
  permuteList.setAll(0, permute_list);

  final resultTensorPtr =
      Tensor_permute(a._tensorPtr, permutePointer, permute_list.length);
  final errorMsg = _get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr);

  return tensor;
}

Tensor sum(Tensor a, {bool has_type = false, int Dtype = float32}) {
  return a.sum(has_type: has_type, Dtype: Dtype);
}

Tensor add(dynamic a, dynamic b, {double alpha = 1}) {
  if (a is num) {
    return add(b, a, alpha: alpha);
  } else if (a is Tensor) {
    return a.add(b, alpha: alpha);
  } else {
    throw Exception("wrong data type");
  }
}

Tensor sub(dynamic a, dynamic b, {double alpha = 1}) {
  if (a is num) {
    return sub(b, a, alpha: alpha);
  } else if (a is Tensor) {
    return a.sub(b, alpha: alpha);
  } else {
    throw Exception("wrong data type");
  }
}

Tensor mul(dynamic a, dynamic b) {
  if (a is num) {
    return mul(b, a);
  } else if (a is Tensor) {
    return a.mul(b);
  } else {
    throw Exception("wrong data type.");
  }
}

Tensor div(dynamic a, dynamic b, {String rounding_mode = ""}) {
  if (a is num) {
    return div(b, a);
  } else if (a is Tensor) {
    return a.div(b, rounding_mode: rounding_mode);
  } else {
    throw Exception("wrong data type.");
  }
}

Tensor add_(dynamic a, dynamic b, {double alpha = 1}) {
  if (a is num) {
    return add_(b, a, alpha: alpha);
  } else if (a is Tensor) {
    if (b is Tensor) {
      final alphaScalar = float64_to_scalar(alpha);

      Tensor_add_(a._tensorPtr, b._tensorPtr, alphaScalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);

      return tensor;
    } else if (b is num) {
      if (b is int) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = int32_to_scalar(b);

        Tensor_add_scalar_(
            a._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else if (b is double) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = float64_to_scalar(b);

        Tensor_add_scalar_(
            a._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  } else {
    throw Exception("wrong data type");
  }
}

Tensor sub_(dynamic a, dynamic b, {double alpha = 1}) {
  if (a is num) {
    return sub_(b, a, alpha: alpha);
  } else if (a is Tensor) {
    if (b is Tensor) {
      final alphaScalar = float64_to_scalar(alpha);

      Tensor_sub_(a._tensorPtr, b._tensorPtr, alphaScalar.scalarPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);

      return tensor;
    } else if (b is num) {
      if (b is int) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = int32_to_scalar(b);

        Tensor_sub_scalar_(
            a._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else if (b is double) {
        final alphaScalar = float64_to_scalar(alpha);
        final rightScalar = float64_to_scalar(b);

        Tensor_sub_scalar_(
            a._tensorPtr, rightScalar.scalarPtr, alphaScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  } else {
    throw Exception("wrong data type");
  }
}

Tensor mul_(dynamic a, dynamic b) {
  if (a is num) {
    return mul_(b, a);
  } else if (a is Tensor) {
    if (b is Tensor) {
      Tensor_mul_(a._tensorPtr, b._tensorPtr);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
    } else if (b is num) {
      if (b is int) {
        final rightScalar = int32_to_scalar(b);

        Tensor_mul_scalar_(a._tensorPtr, rightScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else if (b is double) {
        final rightScalar = float64_to_scalar(b);

        Tensor_mul_scalar_(a._tensorPtr, rightScalar.scalarPtr);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  } else {
    throw Exception("wrong data type.");
  }
}

Tensor div_(dynamic a, dynamic b, {String rounding_mode = ""}) {
  if (a is num) {
    return div_(b, a);
  } else if (a is Tensor) {
    if (b is Tensor) {
      final units = utf8.encode(rounding_mode);
      // 在本地分配足够的内存来复制这个 Uint8List
      final Pointer<Uint8> result =
          malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
      // 获取 Uint8List 的指针
      final Uint8List nativeString = result.asTypedList(units.length + 1);
      // 将 Uint8List 复制到分配的内存中
      nativeString.setRange(0, units.length, units);
      // 确保以 null 字节结尾，满足 C 语言对字符串的要求
      nativeString[units.length] = 0;
      // 返回指向已编码字符串的指针

      final rounding_mode_Utf8 =
          rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();

      Tensor_div_(a._tensorPtr, b._tensorPtr, rounding_mode_Utf8);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final tensor = Tensor._internal(a._tensorPtr);
      return tensor;
    } else if (b is num) {
      if (b is int) {
        final rightScalar = int32_to_scalar(b);

        final units = utf8.encode(rounding_mode);
        // 在本地分配足够的内存来复制这个 Uint8List
        final Pointer<Uint8> result =
            malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
        // 获取 Uint8List 的指针
        final Uint8List nativeString = result.asTypedList(units.length + 1);
        // 将 Uint8List 复制到分配的内存中
        nativeString.setRange(0, units.length, units);
        // 确保以 null 字节结尾，满足 C 语言对字符串的要求
        nativeString[units.length] = 0;
        // 返回指向已编码字符串的指针
        final rounding_mode_Utf8 =
            rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();

        Tensor_div_scalar_(
            a._tensorPtr, rightScalar.scalarPtr, rounding_mode_Utf8);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else if (b is double) {
        final rightScalar = float64_to_scalar(b);
        final units = utf8.encode(rounding_mode);
        // 在本地分配足够的内存来复制这个 Uint8List
        final Pointer<Uint8> result =
            malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
        // 获取 Uint8List 的指针
        final Uint8List nativeString = result.asTypedList(units.length + 1);
        // 将 Uint8List 复制到分配的内存中
        nativeString.setRange(0, units.length, units);
        // 确保以 null 字节结尾，满足 C 语言对字符串的要求
        nativeString[units.length] = 0;
        // 返回指向已编码字符串的指针
        final rounding_mode_Utf8 =
            rounding_mode.isEmpty ? nullptr : result.cast<Utf8>();

        Tensor_div_scalar_(
            a._tensorPtr, rightScalar.scalarPtr, rounding_mode_Utf8);
        final errorMsg = _get_and_reset_last_err();
        if (errorMsg != nullptr) {
          final errorString = errorMsg.cast<Utf8>().toDartString();

          throw Exception(errorString);
        }

        final tensor = Tensor._internal(a._tensorPtr);
        return tensor;
      } else {
        throw Exception("wrong data type");
      }
    } else {
      throw Exception("wrong data type.");
    }
  } else {
    throw Exception("wrong data type.");
  }
}

void save(Tensor a, String path) {
  if (Platform.isWindows || Platform.isLinux) {
    final units = utf8.encode(path);
    // 在本地分配足够的内存来复制这个 Uint8List
    final Pointer<Uint8> result =
        malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
    // 获取 Uint8List 的指针
    final Uint8List nativeString = result.asTypedList(units.length + 1);
    // 将 Uint8List 复制到分配的内存中
    nativeString.setRange(0, units.length, units);
    // 确保以 null 字节结尾，满足 C 语言对字符串的要求
    nativeString[units.length] = 0;
    // 返回指向已编码字符串的指针
    final path_Utf8 = result.cast<Utf8>();

    Tensor_save(a._tensorPtr, path_Utf8);
    final errorMsg = _get_and_reset_last_err();
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }
  } else {
    throw Exception("only support desktop platform");
  }
}

Tensor load(String path) {
  if (Platform.isWindows || Platform.isLinux) {
    final units = utf8.encode(path);
    // 在本地分配足够的内存来复制这个 Uint8List
    final Pointer<Uint8> result =
        malloc.allocate<Uint8>(units.length + 1); // 注意加 1，为了 null 结尾
    // 获取 Uint8List 的指针
    final Uint8List nativeString = result.asTypedList(units.length + 1);
    // 将 Uint8List 复制到分配的内存中
    nativeString.setRange(0, units.length, units);
    // 确保以 null 字节结尾，满足 C 语言对字符串的要求
    nativeString[units.length] = 0;
    // 返回指向已编码字符串的指针
    final path_Utf8 = result.cast<Utf8>();
    final resultTensorPtr = Tensor_load(path_Utf8);
    final errorMsg = _get_and_reset_last_err();

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();

      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr);

    return tensor;
  } else {
    throw Exception("only support desktop platform");
  }
}

Tensor IntTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  List<bool> isFirstElementAtDepth = [];

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElementAtDepth.length <= depth ||
          isFirstElementAtDepth[depth]) {
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
      from_blob(flatList.cast<double>(), sizes, int32, int32);
  return outputTensor;
}

Tensor FloatTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  List<bool> isFirstElementAtDepth = [];

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElementAtDepth.length <= depth ||
          isFirstElementAtDepth[depth]) {
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
if(flatList is Float32List){
  Tensor outputTensor =
      from_blob(flatList.cast<double>(), sizes, float32, float32);
      return outputTensor;
}
else if(flatList is Float64List)
{
   Tensor outputTensor =
      from_blob(flatList.cast<double>(), sizes, float64, float32);
      return outputTensor;
}
else{
   Tensor outputTensor =
      from_blob(flatList.cast<double>(), sizes, float32, float32);
      return outputTensor;
}
  
}

Tensor DoubleTensor(dynamic list) {
  List<num> flatList = [];
  List<int> sizes = [];
  List<bool> isFirstElementAtDepth = [];

  void flatten(dynamic element, int depth) {
    if (element is List) {
      if (isFirstElementAtDepth.length <= depth ||
          isFirstElementAtDepth[depth]) {
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
      from_blob(flatList.cast<double>(), sizes, float64, float64);
  return outputTensor;
}

Scalar int32_to_scalar(int value) {
  final result = _int32_to_scalar(value);
  final errorMsg = _get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }
  final scalar = Scalar(result);
  return scalar;
}

Scalar float32_to_scalar(double value) {
  final result = _float32_to_scalar(value);
  final errorMsg = _get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }
  final scalar = Scalar(result);
  return scalar;
}

Scalar float64_to_scalar(double value) {
  final result = _float64_to_scalar(value);
  final errorMsg = _get_and_reset_last_err();
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();

    throw Exception(errorString);
  }
  final scalar = Scalar(result);
  return scalar;
}
