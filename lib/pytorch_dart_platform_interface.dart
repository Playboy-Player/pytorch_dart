import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'pytorch_dart_method_channel.dart';

abstract class PytorchDartPlatform extends PlatformInterface {
  /// Constructs a PytorchDartPlatform.
  PytorchDartPlatform() : super(token: _token);

  static final Object _token = Object();

  static PytorchDartPlatform _instance = MethodChannelPytorchDart();

  /// The default instance of [PytorchDartPlatform] to use.
  ///
  /// Defaults to [MethodChannelPytorchDart].
  static PytorchDartPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [PytorchDartPlatform] when
  /// they register themselves.
  static set instance(PytorchDartPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
