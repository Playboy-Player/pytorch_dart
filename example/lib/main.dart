import 'dart:ffi';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:pytorch_dart/pytorch_dart.dart' as torch;
import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'package:image/image.dart' as img;
import 'package:file_picker/file_picker.dart';
import 'dart:developer' as dev;
import 'dart:convert';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';



const title = 'Image Classification Example';
var module=torch.jit_load("traced_resnet_model.pt");
late Directory tempDir;
var mean=torch.FloatTensor([0.485, 0.456, 0.406]);
var std=torch.FloatTensor([0.229, 0.224, 0.225]);


void main() {
  WidgetsFlutterBinding.ensureInitialized();
  

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: title,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final _picker = ImagePicker();

  bool _isProcessed = false;
  bool _isWorking = false;

 
Future<List<String>> decodeJson(String filepath)
async{
final file = File(filepath);
 final jsonString = await file.readAsString();

  // 解析JSON内容为List<String>
  final List<dynamic> jsonList = jsonDecode(jsonString);
  
  // 创建一个长度为1000的列表
  List<String> labels = List<String>.filled(1000, '');

  // 填充列表，最多填充1000个标签
  for (int i = 0; i < jsonList.length && i < 1000; i++) {
    labels[i] = jsonList[i];
  }

  
return labels;
}
  Future<String?> pickAnImage() async {
    if (Platform.isIOS || Platform.isAndroid) {
      return _picker
          .pickImage(
            source: ImageSource.gallery,
            imageQuality: 100,
          )
          .then((v) => v?.path);
    } else {
      return FilePicker.platform
          .pickFiles(
            dialogTitle: 'Pick an image',
            type: FileType.image,
            allowMultiple: false,
          )
          .then((v) => v?.files.first.path);
    }
  }

  Future<void> takeImageAndProcess() async {
    
    final imagePath = await pickAnImage();
    final List<String> labelList=await decodeJson("imagenet_labels.json");
    if (imagePath == null) {
      return;
    }
    else{
 final imageData=await loadImageAsUint8List(imagePath);
  final image = img.decodeImage(imageData);
    final int height=image!.height;
    final int width=image!.width;
    // Resizing image fpr model, [300, 300]
    final imageInput = img.copyResize(
      image,
      width: 224,
      height: 224,
    );

    // Creating matrix representation, [300, 300, 3]
    final imageMatrix = List.generate(
  imageInput.height,
  (y) => List.generate(
    imageInput.width,
    (x) {
      final pixel = imageInput.getPixel(x, y);
      return [pixel.b.toDouble(), pixel.g.toDouble(), pixel.r.toDouble()];
    },
  ),
);
//pytorch-like operations
var RawTensor=torch.DoubleTensor(imageMatrix);
RawTensor=RawTensor/256;
RawTensor=(RawTensor-mean)/std;
var inputTensor=RawTensor.permute([2,0,1]).unsqueeze(0);
var outputTensor=module.forward([inputTensor]);
//
if(outputTensor is torch.Tensor)
{
  final prTensor=RawTensor.toList();
  print(prTensor);
  var predRes=outputTensor.topk(1);
  print(predRes);
  final predResult=predRes[1].toList();
  final intList=List<int>.from(predResult);
  print(labelList[intList[0]]);
}



    }
   

  }

  Future<Uint8List> loadImageAsUint8List(String imagePath) async {
 
  File imageFile = File(imagePath);

  if (!imageFile.existsSync()) {
    throw Exception('指定路径的图片文件不存在');
  }

  List<int> imageBytes = await imageFile.readAsBytes();
  Uint8List uint8List = Uint8List.fromList(imageBytes);

  return uint8List;
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: Stack(
        children: <Widget>[
          Center(
            child: ListView(
              shrinkWrap: true,
              children: <Widget>[
                if (_isProcessed && !_isWorking)
                  ConstrainedBox(
                    constraints: BoxConstraints(maxWidth: 3000, maxHeight: 300),
                    
                  ),
                Column(
                  children: [
                    
                    ElevatedButton(
                      child: Text('Process photo'),
                      onPressed: takeImageAndProcess,
                    ),
                  ],
                )
              ],
            ),
          ),
          if (_isWorking)
            Positioned.fill(
              child: Container(
                color: Colors.black.withOpacity(.7),
                child: Center(
                  child: CircularProgressIndicator(),
                ),
              ),
            ),
        ],
      ),
    );
  }
}