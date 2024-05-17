import 'dart:ffi'; // For FFI
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

class Scalar{
  Pointer<Void> _scalarPtr;

  Scalar(this._scalarPtr);
Pointer<Void> get scalarPtr => _scalarPtr;
}