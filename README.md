# pytorch_dart

A dart wrapper for Libtorch,provides some fundametal bindings.
You can use it as an alternative to NumPy.

## Getting Started
1. Add pytorch_dart to your pubspec.yaml
2. ```
    dart run pytorch_dart:setup --platform <your_platform>
    ```
    `<your_platform>` only support `linux` and `android` now.  
    Attention:In setup,you need to have access to Google Drive.
3. ```dart
    import 'package:pytorch_dart/pytorch_dart.dart' as torch;

    ```
4. For Android developers
    libtorch for andoroid needs a specific version of NDK,so [install NDK](https://developer.android.com/studio/projects/install-ndk?hl=zh-cn) and choose version 21.4.7075529  

    Then add a column in 'android/local.properties'  

    ```
    ndk.dir=<path_to_your_ndk>/21.4.7075529
    ```
    After adding a column,your 'local.properties' should look like this:  
    ```
    flutter.sdk=/home/pc/flutter
    sdk.dir=/home/pc/Android/Sdk
    flutter.buildMode=debug
    ndk.dir=/home/pc/Android/Sdk/ndk/21.4.7075529
    ```

## Usage
1. It only include some basic functions about torch.Tensor now
2. The functions avaliable now:  
`torch.empty()`  
`torch.eye()`  
`torch.ones()`  
`torch.arange(double start, double end, double step,{bool requiresGrad = false})`  
`torch.linspace(double start, double end, int steps,{bool requiresGrad = false})`  
`torch.logspace(double start, double end, int steps, double base,{bool requiresGrad = false})`  
`torch.equal(Tensor a,Tensor b)`  
`torch.add(Tensor a, tensor b,{double alpha=1,inplace=false})`  
`torch.sub(Tensor a, tensor b,{double alpha=1,inplace=false})`  
`torch.mul(Tensor a, tensor b,{double alpha=1,inplace=false})`  
`torch.div(Tensor a, tensor b,{double alpha=1,inplace=false})`  
`torch.sum(Tensor a)`  
`torch.mm(Tensor a, Tensor b)`  
3. Example
```dart
torch.Tensor d=torch.eye(3,2);
print(d);
```
Result:
```
flutter:  1  0
 0  1
 0  0
[ CPUFloatType{3,2} ]
```
## Roadmap
1. Add support for other functions such as `torch.permute()`
2. Add support for Windows and iOS
3. Add support for other functions,such as `torch.nn`

## Acknowledgement
This project uses [pytorch-flutter-FFI-example](https://github.com/dvagala/pytorch-flutter-FFI-example) and [gotorch](https://github.com/wangkuiyi/gotorch)