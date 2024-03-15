import 'package:flutter_test/flutter_test.dart';
import 'package:pytorch_dart/pytorch_dart.dart';
import 'package:pytorch_dart/native_pytorch_platform_interface.dart';
import 'package:pytorch_dart/native_pytorch_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockNativePytorchPlatform
    with MockPlatformInterfaceMixin
    implements NativePytorchPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final NativePytorchPlatform initialPlatform = NativePytorchPlatform.instance;

  test('$MethodChannelNativePytorch is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelNativePytorch>());
  });

  test('getPlatformVersion', () async {
    NativePytorch nativePytorchPlugin = NativePytorch();
    MockNativePytorchPlatform fakePlatform = MockNativePytorchPlatform();
    NativePytorchPlatform.instance = fakePlatform;

    expect(await nativePytorchPlugin.getPlatformVersion(), '42');
  });
}
