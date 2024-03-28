import 'package:flutter_test/flutter_test.dart';
import 'package:pytorch_dart/pytorch_dart.dart';
import 'package:pytorch_dart/pytorch_dart_platform_interface.dart';
import 'package:pytorch_dart/pytorch_dart_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockPytorchDartPlatform
    with MockPlatformInterfaceMixin
    implements PytorchDartPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final PytorchDartPlatform initialPlatform = PytorchDartPlatform.instance;

  test('$MethodChannelPytorchDart is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelPytorchDart>());
  });

  test('getPlatformVersion', () async {
    PytorchDart pytorchDartPlugin = PytorchDart();
    MockPytorchDartPlatform fakePlatform = MockPytorchDartPlatform();
    PytorchDartPlatform.instance = fakePlatform;

    expect(await pytorchDartPlugin.getPlatformVersion(), '42');
  });
}
