import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'pytorch_dart_platform_interface.dart';

/// An implementation of [PytorchDartPlatform] that uses method channels.
class MethodChannelPytorchDart extends PytorchDartPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('pytorch_dart');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
