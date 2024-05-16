import 'dart:ffi'; // For FFI
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

// Dart doesn't have direct pinning like C#, but Uint8List provides a mutable buffer.
class PinnedArray {
  Pointer<Pointer<Void>> _pointer=nullptr;
  late Uint8List _array;
  late int _length;

  // Allocate and pin memory
  Pointer<Pointer<Void>> createArray(int length) {
    freeHandle();

    _array = Uint8List(length);
    _pointer = calloc<Pointer<Void>>(length);
    

    return _pointer;
  }

  // Overloaded method for IntPtr length
  Pointer<Pointer<Void>> createArrayFromPointer(Pointer<Int32> length) {
    return createArray(length.value);
  }

  // Overloaded method for existing array
   Pointer<Pointer<Void>> createArrayFromArray(Uint8List array) {
    freeHandle();

    _array = array;
    _pointer = calloc<Pointer<Void>>(array.length);
    //Unfinished

    return _pointer;
  }

  // Cleanup and free allocated memory
  void dispose() {
    freeHandle();
  }

  void freeHandle() {
    if (malloc != null && _pointer != null) {
      calloc.free(_pointer);
      // Set `_pointer` to `nullptr` to avoid accessing freed memory
      _pointer = nullptr;
    }
  }
  Pointer<Pointer<Void>> get pointer => _pointer;
  
}

final PinnedArray allocator=PinnedArray();

Pointer<Pointer<Void>> allocateMemory(int length) {
    return allocator.createArray(length);
  }

// Function pointers for Dart-C interaction
typedef AllocatePinnedArrayNative =  Pointer<Pointer<Void>> Function(Int length);
typedef AllocatePinnedArray =  Pointer<Pointer<Void>> Function(Int length);

