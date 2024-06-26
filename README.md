# Pytorch_Dart

Pytorch_Dart is a dart wrapper for Libtorch,striving to provide an experience identical to [PyTorch](https://github.com/pytorch/pytorch).

You can use it as an alternative to Numpy in your Dart/Flutter projects.

**This package is experimental and APIs may change in the future**.

| Platform | Status | Prebuilt binaries                               |
| -------- | ------ | ----------------------------------------------- |
| Windows  | ✅     | x64(without CUDA)                               |
| Android  | ✅     | arm64-v8a<br />armeabi-v7a<br />x86_64<br />x86 |
| Linux    | ✅     | x64(without CUDA)                               |
| iOS      | ❌     | coming soon                                     |
| MacOS    | ❌     | coming soon                                     |

Theoretically you can run pytorch_dart on MacOS by simply replace `/libtorch-linux/libtorch` with libtorch for MacOS.

## Attention

Pytorch_dart will be refactored based on the C wrapper of TorchSharp since v0.1.1,the last version using the C wrapper of gotorch is 0.1.0

## Getting Started

### Add pytorch_dart to your pubspec.yaml

```dart
    pytorch_dart:^0.1.0
```

### Setup

```dart
dart run pytorch_dart:setup --platform <your_platform>
```

`<your_platform>` only support `linux` , `android` and `windows` now.

### Enjoy it!

```dart
    import 'package:pytorch_dart/pytorch_dart.dart' as torch;

```

### For Android developers

libtorch for android needs a specific version of NDK,so [install NDK](https://developer.android.com/studio/projects/install-ndk?hl=zh-cn) and choose version 21.4.7075529

Then add a row in `android/local.properties` in your project

```dart
ndk.dir=<path_to_your_ndk>/21.4.7075529
```

After adding a row,your `local.properties` should look like this:

```gradle
flutter.sdk=/home/pc/flutter
sdk.dir=/home/pc/Android/Sdk
flutter.buildMode=debug
ndk.dir=/home/pc/Android/Sdk/ndk/21.4.7075529
```

Also,'torch.load()' and 'torch.save()' are not available on Android.In fact,they only work well on Linux

Some exceptions they threw in Windows are like this:

```
[ERROR:flutter/runtime/dart_vm_initializer.cc(41)] Unhandled Exception: Exception: [enforce fail at inline_container.cc:633] . invalid file name: 
00007FFAA49D3F08 <unknown symbol address> c10.dll!<unknown symbol> [<unknown file> @ <unknown line number>]
00007FFAA49D4823 <unknown symbol address> c10.dll!<unknown symbol> [<unknown file> @ <unknown line number>]
```

### For windows developers

#### Troubleshooting

```
Launching lib\main.dart on Windows in debug mode...
√  Built build\windows\x64\runner\Debug\example.exe.
Error waiting for a debug connection: The log reader stopped unexpectedly, or never started.
Error launching application on Windows.
```

#### Solutions:

1. Download libtorch from [here](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.2%2Bcpu.zip)
2. Unzip it
3. copy all the files from `libtorch\lib\` to `build\windows\x64\runner\Debug\`

## Usage

### Brief Introduction

1. It include some basic functions in [torch](https://pytorch.org/docs/stable/torch.html) now.
2. Support for [torch.nn](https://pytorch.org/docs/stable/nn.html) is coming soon.
3. **Almost all function usages remain consistent with PyTorch.**
4. **Broadcasting also works for pytorch_dart.**
5. Example

```dart
var d=torch.eye(3,2);
print(d);
```

Result:

```
flutter:
 1  0
 0  1
 0  0
[ CPUFloatType{3,2} ]
```

### Operator overloading

Attention:Dart has no magic functions(like `_radd_` in python).Therefore, in binary operators, tensor can only be on the left side.

Example

```dart
import 'package:pytorch_dart/pytorch_dart.dart' as torch;
...

var c=torch.DoubleTensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]);
var d=c+10;// no errors
var e=10+c;//cause errors
```

Other binary operators (`-`,`*`,`/`)are just like `+`

For operator `[]` ,you can use it just like in Pytorch.

However,in current version,slicing is not supported.Therefore,you cant't use `[a:b]` to select sub tensor.

Example

```dart
import 'package:pytorch_dart/pytorch_dart.dart' as torch;
...

var c=torch.DoubleTensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]);
print(c[0][0]);
```

Result

```dart
flutter: 1
[ CPUDoubleType{} ]
```

### torch

1. `torch.tensor()` is not supported in pytorch_dart,use `torch.IntTensor()`,`torch.FloatTensor` or `torch.DoubleTensor` to create tensors.
2. Functions avaliable now:
   `torch.empty()`
   `torch.eye()`
   `torch.ones()`
   `torch.IntTensor(List<int> list)`
   `torch.FloatTensor(List<double> list)`
   `torch.DoubleTensor(List<double> list)`
   `torch.arange(double start, double end, double step,{bool requiresGrad = false})`
   `torch.linspace(double start, double end, int steps,{bool requiresGrad = false})`
   `torch.logspace(double start, double end, int steps, double base,{bool requiresGrad = false})`
   `torch.equal(Tensor a,Tensor b)`
   `torch.add(Tensor a, tensor b,{double alpha=1})`
   `torch.sub(Tensor a, tensor b,{double alpha=1})`
   `torch.mul(Tensor a, tensor b)`
   `torch.div(Tensor a, tensor b)`
   `torch.add_(Tensor a, tensor b,{double alpha=1})`
   `torch.sub_(Tensor a, tensor b,{double alpha=1})`
   `torch.mul_(Tensor a, tensor b)`
   `torch.div_(Tensor a, tensor b)`
   `torch.sum(Tensor a)`
   `torch.mm(Tensor a, Tensor b)`
   `torch.transpose(Tensor a,int dim0,int dim1)`
   `torch.permute(Tensor a,List <int> permute_list)`
   `torch.save(Tensor a,String path)`
   `torch.load(String path)`
3. Almost all function usages remain consistent with PyTorch.
4. Some in-place operation are supported,such as `torch.add_()`
5. Example

   ```dart
   import 'package:pytorch_dart/pytorch_dart.dart' as torch;
   ...

   var c=torch.DoubleTensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]);
   var d=torch.add(10,c)
   print(d)
   ```

   Result:

   ```dart
   flutter:
    11  12  13
    14  15  16
   [ CPUDoubleType{2,3} ]
   ```

### torch.tensor

1. Functions avaliable now:
   `.dim()`
   `.dtype()`
   `.shape()`
   `.size()`
   `.detach()`
   `.add_()`
   `.sub_()`
   `.mul_()`
   `.div_()`
   `.toList()`
2. `.dtype()` is different from its implementation in Pytorch.

   In Pytorch,`.dtype` returns an object represents the data type of a tensor

   But in pytorch_dart,`.dtype()` returns a number represents the data type of a tensor.(maybe I will rewrite it later)

   Example

   ```dart
   import 'package:pytorch_dart/pytorch_dart.dart' as torch;
   ...

   var c=torch.DoubleTensor([[1.0,2.0,3.0],[4.0,5.0,6.0]]);
   print(c.dtype())
   ```

   Result

   ```dart
   flutter: 7
   ```

   `7` represents `torch.float64.`

   All the corresponding relations are in `lib/src/constants.dart`
3. Other function usages remain consistent with PyTorch.

## Roadmap

1. Add support for iOS and MacOS.
2. Add support for other functions,such as `torch.nn`

## Acknowledgement

This project uses [pytorch-flutter-FFI-example](https://github.com/dvagala/pytorch-flutter-FFI-example) and [gotorch](https://github.com/wangkuiyi/gotorch)
