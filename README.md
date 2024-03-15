# pytorch_dart

A dart wrapper for Libtorch,provides some fundametal bindings.

## Getting Started
1. Add pytorch_dart to your pubspec.yaml
2. ```
    dart run pytorch_dart_setup
    ```
3. ```dart
    import 'package:pytorch_dart/pytorch_dart.dart' as torch;

    ```
## Usage
1. It only include some basic functions about torch.Tensor now
2. Dart code：
    ```dart
    torch.Tensor tensor = torch.createTensorFromList(
      [[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]]
    );
    List<dynamic> listOrig = torch.createListFromTensor(tensor);
    torch.Tensor t = tensor.transpose( 0, 1);
    List<dynamic> listTranspose = torch.createListFromTensor(t);

    torch.Tensor t1 = torch.createTensorFromList([
      [1.1, 2.2],
      [3.3, 4.4]
    ]);
    torch.Tensor t2 = torch.createTensorFromList([
      [0.1, 0.2],
      [0.3, 0.4]
    ]);
    torch.Tensor t3 = torch.sub(t1, t2);
    torch.Tensor t4 = torch.add(t1, t2);
    List<dynamic> list1 = torch.createListFromTensor(t1);
    List<dynamic> list2 = torch.createListFromTensor(t2);
    List<dynamic> list3 = torch.createListFromTensor(t3);
    List<dynamic> list4 = torch.createListFromTensor(t4);
    print("original Tensor=$listOrig");
    print("transposed Tensor=$listTranspose");
    print("Tensor t1=$list1");
    print("Tensor t2=$list2");
    print("Tensor t3=t1-t2=$list3");
    print("Tensor t4=t1+t2=$list4");

    ```
3. Corresponding python code:
```python
import torch;
tensor=torch.Tensor([[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]])
t=tensor.transpose(0,1)
t1=torch.Tensor([
      [1.1, 2.2],
      [3.3, 4.4]
    ])
t2=torch.Tensor([
      [0.1, 0.2],
      [0.3, 0.4]
    ])
t3=torch.sub(t1,t2)
t4=torch.add(t1,t2)
print(t1)
print(t2)
print(t3)
print(t4)
```
## Roadmap
1. Add support for other functions such as `torch.permute()`
2. Add support for Windows
3. Add support for other functions,such as `torch.nn`

## Acknowledgement
This project uses [pytorch-flutter-FFI-example](https://github.com/dvagala/pytorch-flutter-FFI-example)