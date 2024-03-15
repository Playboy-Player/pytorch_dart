import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'native_pytorch_method_channel.dart';

abstract class NativePytorchPlatform extends PlatformInterface {
  /// Constructs a NativePytorchPlatform.
  NativePytorchPlatform() : super(token: _token);

  static final Object _token = Object();

  static NativePytorchPlatform _instance = MethodChannelNativePytorch();

  /// The default instance of [NativePytorchPlatform] to use.
  ///
  /// Defaults to [MethodChannelNativePytorch].
  static NativePytorchPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [NativePytorchPlatform] when
  /// they register themselves.
  static set instance(NativePytorchPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
