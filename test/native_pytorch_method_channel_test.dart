import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pytorch_dart/native_pytorch_method_channel.dart';

void main() {
  MethodChannelNativePytorch platform = MethodChannelNativePytorch();
  const MethodChannel channel = MethodChannel('native_pytorch');

  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    channel.setMockMethodCallHandler((MethodCall methodCall) async {
      return '42';
    });
  });

  tearDown(() {
    channel.setMockMethodCallHandler(null);
  });

  test('getPlatformVersion', () async {
    expect(await platform.getPlatformVersion(), '42');
  });
}
