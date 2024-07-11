import 'dart:ffi'; // For FFI
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:pytorch_dart/pytorch_dart.dart';
import 'dart:io';
import 'tensor.dart';
import 'dart:convert';
import 'device.dart';

final Pointer<Utf8> Function() _get_and_reset_last_err = nativeLib
    .lookup<NativeFunction<Pointer<Utf8> Function()>>(
        'THSTorch_get_and_reset_last_err')
    .asFunction();

final Pointer<Void> Function(Pointer<Utf8>,int device,int index) JIT_load = nativeLib
    .lookup<NativeFunction<Pointer<Void> Function(Pointer<Utf8>,Int64,Int64)>>(
        'THSJIT_load')
    .asFunction();

final void Function(Pointer<Void> module,Pointer<Void> tRefsHandle,int length,Pointer<NativeFunction<AllocateNativeTensorOrScalarIndexedArray>>,Pointer<Int8> typeCode,int index) JIT_Module_forward = nativeLib
    .lookup<NativeFunction<Void Function(Pointer<Void>,Pointer<Void>,Int,Pointer<NativeFunction<AllocateNativeTensorOrScalarIndexedArray>>,Pointer<Int8> typeCode,Int32)>>(
        'THSJIT_Module_forward')
    .asFunction();
typedef AllocateTensorOrScalarArrayC = Pointer<Void> Function(Int32 length);
typedef AllocateTensorOrScalarArrayDart = Pointer<Void> Function(int length);

typedef GetTensorOrScalarC = Pointer<TensorOrScalar> Function(Pointer<Void> handle, Int32 index);
typedef GetTensorOrScalarDart = Pointer<TensorOrScalar> Function(Pointer<Void> handle, int index);

typedef FreeTensorOrScalarArrayC = Void Function(Pointer<Void> handle);
typedef FreeTensorOrScalarArrayDart = void Function(Pointer<Void> handle);

//use Pointer<Void> to store a list of TensorOrScalar,different from Tensor(may rewrite it later)
final Pointer<Void> Function(int size) allocateTensorOrScalarArray = nativeLib
        .lookup<NativeFunction<Pointer<Void> Function(Int32)>>(
        'THSJIT_AllocateTensorOrScalarArray')
    .asFunction();
    final Pointer<TensorOrScalar> Function(Pointer<Void> tensorOrScalar,int size) getTensorOrScalar = nativeLib
        .lookup<NativeFunction<Pointer<TensorOrScalar> Function(Pointer<Void>,Int32)>>(
        'THSJIT_GetTensorOrScalar')
    .asFunction();
    final Pointer<Void> Function(Pointer<Void> handle) freeTensorOrScalarArray = nativeLib
        .lookup<NativeFunction<Pointer<Void> Function(Pointer<Void>)>>(
        'THSJIT_FreeTensorOrScalarArray')
    .asFunction();


    final void Function(Pointer<Void> array,int index,int typeCode,int arrayIndex,Pointer<Void> handle) setTensorOrScalar = nativeLib
        .lookup<NativeFunction<Void Function(Pointer<Void> array,Int32 index,Int64 typeCode,Int64 arrayIndex,Pointer<Void> handle)>>(
        'THSJIT_SetTensorOrScalar')
    .asFunction();
  class JITModule{

  Pointer<Void> _JITmodulePtr;
  JITModule(this._JITmodulePtr);
  Pointer<Void> get JITmodulePtr=>_JITmodulePtr;


dynamic forward(List<dynamic> input){
var ntosArray =tensorOrScalarAllocator;
var tRefsHandle = DetermineArgumentTypeRefs(input, allocator: ntosArray);

                        var allocated = ntosArray.count;
                        Pointer<Int8> typeCodePtr=calloc<Int8>();
                        JIT_Module_forward(_JITmodulePtr, tRefsHandle,input.length, Pointer.fromFunction<AllocateNativeTensorOrScalarIndexedArray>(allocateTensorOrScalarsMemory),typeCodePtr,allocated);
                         
                         final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
      int typeCode=typeCodePtr.value;
                        List<TensorOrScalar> ptrArray = ntosArray.toTOSArray(allocated);

                        return ProcessReturnValue(ptrArray, typeCode,allocator: ntosArray);


}
  }

JITModule jit_load(String filename,{Device? device_used})
{device_used ??= device("cpu");
   final units = utf8.encode(filename);
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

      final filename_Utf8 =
           result.cast<Utf8>();
           
  final resultModulePtr =
          JIT_load(filename_Utf8,device_used.device_type,device_used.device_index);
      final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }

      final module=JITModule(resultModulePtr);
       malloc.free(result);
      return module;
}





final class TensorOrScalar extends Struct {
 @Int64()
  external int typeCode;

  @Int64()
  external int arrayIndex;
  
   @Int64()
  external int handle;
}


dynamic ProcessReturnValue( List<TensorOrScalar> ptrArray, int typeCode,{ NativeTensorOrScalarIndexedArray ?allocator})
                {allocator??=tensorOrScalarAllocator;
                    switch (typeCode) {
                    
                    case 1:
                        // Tensor
                        final Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[0].handle);
                        return Tensor(handlePointer);
                    case 2:
                        // Tuple
                        switch (ptrArray.length) {
                        case 1:
                        final Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[0].handle);
                            return Tensor(handlePointer);
                        case 2:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                            return ( Tensor(handlePointer0),  Tensor(handlePointer1));
                        case 3:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                        final Pointer<Void> handlePointer2 = Pointer.fromAddress(ptrArray[2].handle);
                            return ( Tensor(handlePointer0),  Tensor(handlePointer1),  Tensor(handlePointer2));
                        case 4:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                        final Pointer<Void> handlePointer2 = Pointer.fromAddress(ptrArray[2].handle);
                        final Pointer<Void> handlePointer3 = Pointer.fromAddress(ptrArray[3].handle);
                            return ( Tensor(handlePointer0),  Tensor(handlePointer1),  Tensor(handlePointer2),  Tensor(handlePointer3));
                        case 5:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                        final Pointer<Void> handlePointer2 = Pointer.fromAddress(ptrArray[2].handle);
                        final Pointer<Void> handlePointer3 = Pointer.fromAddress(ptrArray[3].handle);
                        final Pointer<Void> handlePointer4 = Pointer.fromAddress(ptrArray[4].handle);
                            return ( Tensor(handlePointer0),  Tensor(handlePointer1),  Tensor(handlePointer2),  Tensor(handlePointer3),  Tensor(handlePointer4));
                        default: {
                                // Too long a tuple, return as a list, instead.
                              List<Tensor> result =[];
                                for (var i = 0; i < ptrArray.length; i++) {
                                 Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[i].handle);
                       
                                    result.add(Tensor(handlePointer));
                                }
                                return result;
                            }
                        }
                    case 3: {
                            // List of tensors
                            List<Tensor> result =[];
                                for (var i = 0; i < ptrArray.length; i++) {
                                  Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[i].handle);
                                    result.add(Tensor(handlePointer));
                                }
                                return result;
                        }
                    case 4:
                        // Scalar
                        final Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[0].handle);
                        return  Scalar(handlePointer);
                    case 5:
                        // Scalar tuple
                        switch (ptrArray.length) {
                        case 1:
                        final Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[0].handle);
                            return Scalar(handlePointer);
                        case 2:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                            return ( Scalar(handlePointer0),  Scalar(handlePointer1));
                        case 3:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                        final Pointer<Void> handlePointer2 = Pointer.fromAddress(ptrArray[2].handle);
                            return ( Scalar(handlePointer0),  Scalar(handlePointer1),  Scalar(handlePointer2));
                        case 4:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                        final Pointer<Void> handlePointer2 = Pointer.fromAddress(ptrArray[2].handle);
                        final Pointer<Void> handlePointer3 = Pointer.fromAddress(ptrArray[3].handle);
                            return ( Scalar(handlePointer0),  Scalar(handlePointer1),  Scalar(handlePointer2),  Scalar(handlePointer3));
                        case 5:
                        final Pointer<Void> handlePointer0 = Pointer.fromAddress(ptrArray[0].handle);
                        final Pointer<Void> handlePointer1 = Pointer.fromAddress(ptrArray[1].handle);
                        final Pointer<Void> handlePointer2 = Pointer.fromAddress(ptrArray[2].handle);
                        final Pointer<Void> handlePointer3 = Pointer.fromAddress(ptrArray[3].handle);
                        final Pointer<Void> handlePointer4 = Pointer.fromAddress(ptrArray[4].handle);
                            return ( Scalar(handlePointer0),  Scalar(handlePointer1),  Scalar(handlePointer2),  Scalar(handlePointer3),  Scalar(handlePointer4));
                        default: {
                                // Too long a tuple, return as a list, instead.
                               List<Scalar> result =[];
                                for (var i = 0; i < ptrArray.length; i++) {
                                  Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[i].handle);
                                    result.add(Scalar(handlePointer));
                                }
                                return result;
                            }
                        }
                    case 6: {
                            // List of scalars
                           List<Scalar> result =[];
                            for (var i = 0; i < ptrArray.length; i++) {
                              Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[i].handle);
                                result.add(Scalar(handlePointer));
                            }
                            return result;
                        }
                    case 7: {
                            // List of scalars and tensors
                            var result = <dynamic>[ptrArray.length];
                            for (var i = 0; i < ptrArray.length; i++) {
                                switch (ptrArray[i].typeCode) {
                                case 0:
                                Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[i].handle);
                                    result[i] = Tensor(handlePointer);
                                    break;
                                case 8:
                                    result[i] = null;
                                    break;
                                case 4:
                                 Pointer<Void> handlePointer = Pointer.fromAddress(ptrArray[i].handle);
                                    result[i] =  Scalar(handlePointer);
                                    break;
                                default:
                                    throw Exception("returning something else than a tensor/scalar, a tuple of tensors/scalars, or list of tensors/scalars.");
                                }
                            }
                            return result;
                        }
                    case 8:
                        // The value 'null' of any reference type
                        return null;

                    default:
                        // Nothing.
                        throw Exception("returning something else than a tensor, a tuple of tensors, or list of tensors.");
                    }
                    
                }
Pointer<Void> DetermineArgumentTypeRefs(List<dynamic> list, {NativeTensorOrScalarIndexedArray ?allocator}){
allocator ??=tensorOrScalarAllocator;
final int count=list.length;
var tensorRefs = allocator.createArray(allocator.count, list.length);
for (var idx = 0; idx < list.length; idx++) {
                        switch (list[idx]) {
                        case Tensor t:
                            setTensorOrScalar(tensorRefs, idx, 0, 0, t.tensorPtr);
                            break;
                        case Scalar s:
                            setTensorOrScalar(tensorRefs, idx, 1, 0, s.scalarPtr);
                            break;
                        case List<Tensor> tensors: {
                                setTensorOrScalar(tensorRefs, idx, 5, tensors.length, DetermineArgumentTypeRefs(tensors,allocator:allocator));
                            }
                            break;
                        default:
                            if (list[idx] == null) {
                                setTensorOrScalar(tensorRefs, idx, 8, 0, nullptr);
                            } else {
                                throw Exception("Passing wrong arguments to Pytorch_dart.");
                            }
                            break;
                        }


}
return tensorRefs;
}


class NativeTensorOrScalarIndexedArray {
  final List<Pointer<Void>> _arrays = [];//It represents a list of TensorOrScalar
  final List<int> _sizes = [];
  
  int _allocated = 0;

  

  int get count => _arrays.length;

  Pointer<Void> operator [](int index) => _arrays[index];
//insert a array of TensorOrScalar into _arrays
  void setArrayContent(int index, Pointer<Void> value, int size) {
    _extendHandlesList(index);
    _arrays[index] = value;
    _sizes[index] = size;
  }

  Pointer<Void> createArray(int index, int length) {
    
    final result = allocateTensorOrScalarArray(length);
    if (result == nullptr) {
      throw Exception('Failed to allocate array');
    }
    _allocated += 1;
    setArrayContent(index, result, length);
    return result;
  }

  TensorOrScalar toTOS(Pointer<Void> array, int index) {
    
    final Pointer<TensorOrScalar> ptr = getTensorOrScalar(array, index);
    final ptrRef=ptr.ref;
    
   
     final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
      var resultPtr = calloc<TensorOrScalar>();
      var result=resultPtr.ref;
            result.handle = ptrRef.handle;
            result.arrayIndex = ptrRef.arrayIndex;
            result.typeCode = ptrRef.typeCode;
    return result;
  }

  List<TensorOrScalar> toTOSArray(int index) {
    final size = _sizes[index];
    final ptr = _arrays[index];
   
    final result = <TensorOrScalar>[];
    for (var i = 0; i < size; i++) {
      result.add(toTOS(ptr, i));
      
    }
 final errorMsg = _get_and_reset_last_err();
      if (errorMsg != nullptr) {
        final errorString = errorMsg.cast<Utf8>().toDartString();

        throw Exception(errorString);
      }
    return result;
  }

  void _extendHandlesList(int idx) {
    if (idx >= _arrays.length) {
      final extras = idx - _arrays.length + 1;
      for (var i = 0; i < extras; i++) {
        _arrays.add(nullptr);
        _sizes.add(0);
      }
    }
  }

  void _freeHandles() {
    for (final handle in _arrays) {
      if (handle != nullptr) {
        freeTensorOrScalarArray(handle);
        _allocated -= 1;
      }
    }
    _arrays.clear();
    assert(_allocated == 0);
  }

  @override
  void dispose() {
    _freeHandles();
  }

  @override
  void finalize() {
    _freeHandles();
  }
}


final NativeTensorOrScalarIndexedArray tensorOrScalarAllocator=NativeTensorOrScalarIndexedArray();

Pointer<Void> allocateTensorOrScalarsMemory(int index,int length) {
    return tensorOrScalarAllocator.createArray(index,length);
  }


  typedef AllocateNativeTensorOrScalarIndexedArray =  Pointer<Void> Function(Int index,Int length);