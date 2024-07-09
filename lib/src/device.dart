import 'dart:ffi'; // For FFI
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

class Device {
  int _device_type = 0;
  int _device_index = 0;
  static final Set<int> _usedIndices = {};  // 静态集合，用以记录所有被使用的索引

  // 私有的命名构造函数，仅类内部使用
  Device._internal(this._device_type, this._device_index);

  // 工厂构造函数，用于创建具有适当索引的新设备实例
  factory Device(int deviceType) {
    int newIndex = 0; // 从0开始搜索未使用的最小索引
    while (_usedIndices.contains(newIndex)&&newIndex!=0) {
      newIndex++;  // 如果当前索引已使用，增加索引号
    }
    _usedIndices.add(newIndex);  // 标记为已使用
    return Device._internal(deviceType, newIndex);
  }

  int get device_type => _device_type;
  int get device_index => _device_index;
}

Device device(String type) {
  if (type == "cpu") {
    return Device(0);  // 传递设备类型的标识
  } else {
    throw Exception("Unsupported type");
  }
}

Device defaultDevice=device("cpu");