import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'native_pytorch_platform_interface.dart';

/// An implementation of [NativePytorchPlatform] that uses method channels.
class MethodChannelNativePytorch extends NativePytorchPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('native_pytorch');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
