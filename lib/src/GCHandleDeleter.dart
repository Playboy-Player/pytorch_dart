import 'dart:ffi'; // For FFI
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// 枚举类型，用于标记内存分配方式
  


class Deleter{
  

  // 释放内存，同时根据分配方式选择合适的释放函数
 void _deleteMemory(Pointer<Void> ptr) {
    malloc.free(ptr);
  }
}

final Deleter deleter=Deleter();
void deleteMemory(Pointer<Void> ptr) {
    return deleter._deleteMemory(ptr);
  }

// Function pointers for Dart-C interaction
typedef DeleterNative =  Void Function(Pointer<Void> ptr);
