import 'package:flutter/material.dart';

import 'package:pytorch_dart/pytorch_dart.dart' as torch;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Print to Text Field'),
        ),
        body: PrintToTextField(),
      ),
    );
  }
}

class PrintToTextField extends StatefulWidget {
  @override
  _PrintToTextFieldState createState() => _PrintToTextFieldState();
}

class _PrintToTextFieldState extends State<PrintToTextField> {
  final TextEditingController _controller = TextEditingController();

  // Custom log function
  void customLog(String message) {
    // Add the message to the text controller
    setState(() {
      _controller.text += "$message\n";
    });
    // Optionally, you can still use developer.log to see logs in the console
    // developer.log(message);
  }

  void _tensorTest() async {
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
    customLog("original Tensor=$listOrig");
    customLog("transposed Tensor=$listTranspose");
    customLog("Tensor t1=$list1");
    customLog("Tensor t2=$list2");
    customLog("Tensor t3=t1-t2=$list3");
    customLog("Tensor t4=t1+t2=$list4");
    setState(() {
     
    });
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          TextField(
            controller: _controller,
            maxLines: null, // Allow the text field to display multiple lines
            readOnly: true, // Set the text field to read-only
          ),
          SizedBox(height: 16.0),
          ElevatedButton(
            onPressed: () {
              _tensorTest();
            },
            child: Text("Test"),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
