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
import 'package:flutter/services.dart' show rootBundle;


const title = 'Image Classification Example';
torch.JITModule? module;
var mean = torch.FloatTensor([0.485, 0.456, 0.406]);
var std = torch.FloatTensor([0.229, 0.224, 0.225]);
Uint8List? imageData;
String? label;

void _loadModel() async{
  module=await torch.jit_load('assets/traced_resnet_model.pt');
}
void main() {
  WidgetsFlutterBinding.ensureInitialized();
_loadModel();
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

  
  Future<List<String>> decodeJson(String filepath) async {
    String jsonString=await rootBundle.loadString(filepath);

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
    final List<String> labelList = await decodeJson('assets/imagenet_labels.json');
    DateTime startTime;
    startTime = DateTime.now();
    if (imagePath == null) {
      return;
    } else {
      imageData = await loadImageAsUint8List(imagePath);
      final image = img.decodeImage(imageData!);
      // Resizing image fpr model, [300, 300]
      final imageInput = img.copyResize(
        image!,
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
      var RawTensor = torch.FloatTensor(imageMatrix);
      RawTensor = RawTensor / 256;
      RawTensor = (RawTensor - mean) / std;
      var inputTensor = RawTensor.permute([2, 0, 1]).unsqueeze(0);
      var outputTensor = module!.forward([inputTensor]);
//
      if (outputTensor is torch.Tensor) {
        var predRes = outputTensor.topk(1);
        final predResult = predRes[1].toList();
        final intList = List<int>.from(predResult);
        dev.log(labelList[intList[0]]);
        label=labelList[intList[0]];
           DateTime endTime = DateTime.now();
  Duration elapsedTime = endTime.difference(startTime);
  dev.log('Function took ${elapsedTime.inMilliseconds} milliseconds to execute.');
        setState(() {
    
  });
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
            
                
                child:Column(
                  children: [
                    Expanded(
                      child: Center(
                        child: (imageData != null)
                            ? Image.memory(imageData!)
                            : Container(),
                      ),
                    ),
                    ElevatedButton(
                      child: Text('Select photo'),
                      onPressed: takeImageAndProcess,
                    ),
                 Expanded(
                      child: Center(
                        child: (label != null)
                            ? Text("Predicted result:$label")
                            : Container(),
                      ),
                    ),
                  
                  ],
                ),
             
          ),
          
            
        ],
      ),
    );
  }
}
