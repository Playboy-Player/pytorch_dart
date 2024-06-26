import 'dart:ffi';

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
    torch.Tensor a=torch.empty([2,3]);
    torch.Tensor b=torch.ones([2,2,3],dtype: torch.float64);
    print(b);
   
    
     torch.Tensor c=torch.DoubleTensor([[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],[[10.0,11.0,12.0],[13.0,14.0,15.0],[16.0,17.0,18.0]]]);
     var i=torch.add(10,c);
 
    
    torch.Tensor d=torch.eye(3,3);
   var e=torch.from_blob([1,2,3,4,5,6],[2,3],torch.int32,torch.float64);
   print(e.dtype());
    
   
    
    
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
