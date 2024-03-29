
import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'dart:developer' as dev;

final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libpytorch_dart.so')
    : DynamicLibrary.process();

final Pointer<Utf8> Function(Pointer<Int64> size, int length, int requiresGrad,
        Pointer<Pointer<Void>> result) _empty =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(
                    Pointer<Int64> size,
                    Int64 length,
                    Int64 requiresGrad,
                    Pointer<Pointer<Void>> result)>>('Empty')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Int64> size, int length, int requiresGrad,
        Pointer<Pointer<Void>> result) _ones =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Int64> size, Int64 length,
                    Int64 requiresGrad, Pointer<Pointer<Void>> result)>>('Ones')
        .asFunction();
final Pointer<Utf8> Function(
        int n, int m, int requiresGrad, Pointer<Pointer<Void>> result) _eye =
    nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Int64 n, Int64 m, Int64 requiresGrad,
                    Pointer<Pointer<Void>> result)>>('Eye')
        .asFunction();

final Pointer<Utf8> Function(Pointer<Void>) _print = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function(Pointer<Void>)>>('Tensor_Print')
    .asFunction();

final Pointer<Utf8> Function(Pointer<Void>, Pointer<Pointer<Void>> result)
    _detach = nativeLib
        .lookup<
            NativeFunction<
                Pointer<Utf8> Function(Pointer<Void> tensor,
                    Pointer<Pointer<Void>> result)>>('Tensor_Detach')
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


// 类定义
class TypedNumberList<T extends num> {
  List<T> _values = [];

  TypedNumberList([List<T>? values]) {
    if (values != null) {
      _values = List<T>.from(values);
    }
  }
  void add(T value) {
    _values.add(value);
  }

  List<T> getValues() {
    return _values;
  }

  TypedNumberList.fromList(List<T> values) {
    _values = List<T>.from(values);
  }
}

TypedData convertToTypedNumberList<T extends num>(TypedNumberList<T> list) {
  if (T == int) {
    // 如果T是int类型，返回Int32List
    return Int32List.fromList(list.getValues().cast<int>());
  } else if (T == double) {
    // 如果T是double类型，返回Float64List
    return Float64List.fromList(list.getValues().cast<double>());
  } else {
    throw ArgumentError('Unsupported type T', 'T');
  }
}

class Tensor {
  Pointer<Void> _tensorPtr;

  Tensor._internal(this._tensorPtr);

  @override
  String toString() {
    
    var stringPtr=_print(_tensorPtr);
    final string=stringPtr.cast<Utf8>().toDartString();
    return string;
  }

  Tensor detach() {
    // 将 Dart 的数组转换为原生指针

    // 调用 C++ 的 Empty 函数
    final resultTensorPtr = calloc<Pointer<Void>>();

    final errorMsg = _detach(_tensorPtr, resultTensorPtr);

    // 释放原生数组内存

    // 检查是否有错误信息，如果有，则抛出异常
    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr); // 释放结果指针

    return tensor;
  }
}

Tensor from_blob<T extends num>(TypedNumberList<T> list, List<int> sizes_data) {
  if (T == int) {
    var intList = convertToTypedNumberList(list) as Int32List;
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
        dataPointer.cast(), 3, sizesPointer, sizesList.length, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr.value);
    return tensor;
  } else if (T == double) {
    var floatList = convertToTypedNumberList(list) as Float64List;
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
        dataPointer.cast(), 7, sizesPointer, sizesList.length, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }
    final tensor = Tensor._internal(resultTensorPtr.value);
    return tensor;
  } else {
    throw Exception("wrong type");
  }

  // 使用完毕后释放指针
}

Tensor empty(List<int> size, {bool requiresGrad = false}) {
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _empty(int64Pointer, size.length, requiresGrad ? 1 : 0, resultTensorPtr);

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    calloc.free(errorMsg);
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr); // 释放结果指针

  return tensor;
}

Tensor ones(List<int> size, {bool requiresGrad = false}) {
  // 将 Dart 的数组转换为原生指针
  final Pointer<Int64> int64Pointer = calloc<Int64>(size.length);
  final Int64List int64List = int64Pointer.asTypedList(size.length);
  int64List.setAll(0, size);

  // 调用 C++ 的 Empty 函数
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg =
      _ones(int64Pointer, size.length, requiresGrad ? 1 : 0, resultTensorPtr);

  // 释放原生数组内存
  calloc.free(int64Pointer);

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    calloc.free(errorMsg);
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr); // 释放结果指针

  return tensor;
}

Tensor eye(int n, int m, {bool requiresGrad = false}) {
  final resultTensorPtr = calloc<Pointer<Void>>();
  final errorMsg = _eye(n, m, requiresGrad ? 1 : 0, resultTensorPtr);

  // 释放原生数组内存

  // 检查是否有错误信息，如果有，则抛出异常
  if (errorMsg != nullptr) {
    final errorString = errorMsg.cast<Utf8>().toDartString();
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
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
    calloc.free(errorMsg);
    throw Exception(errorString);
  }

  final tensor = Tensor._internal(resultTensorPtr.value);
  calloc.free(resultTensorPtr);

  return tensor;
}

Tensor add(Tensor a, Tensor b, {double alpha = 1, bool inplace = false}) {
  if (inplace) {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _add_(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  } else {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _add(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  }
}

Tensor sub(Tensor a, Tensor b, {double alpha = 1, bool inplace = false}) {
  if (inplace) {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _sub_(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  } else {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _sub(a._tensorPtr, b._tensorPtr, alpha, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  }
}

Tensor mul(Tensor a, Tensor b, {bool inplace = false}) {
  if (inplace) {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _mul_(a._tensorPtr, b._tensorPtr, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  } else {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _mul(a._tensorPtr, b._tensorPtr, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  }
}

Tensor div(Tensor a, Tensor b, {bool inplace = false}) {
  if (inplace) {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _div_(a._tensorPtr, b._tensorPtr, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  } else {
    final resultTensorPtr = calloc<Pointer<Void>>();
    final errorMsg = _div(a._tensorPtr, b._tensorPtr, resultTensorPtr);

    if (errorMsg != nullptr) {
      final errorString = errorMsg.cast<Utf8>().toDartString();
      calloc.free(errorMsg);
      throw Exception(errorString);
    }

    final tensor = Tensor._internal(resultTensorPtr.value);
    calloc.free(resultTensorPtr);

    return tensor;
  }
}


