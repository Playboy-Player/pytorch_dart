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
    torch.Tensor a=torch.empty([3,2]);
    torch.Tensor b=torch.ones([2,2,3]);
    
   
    print(b);
   
    torch.Tensor c=torch.from_blob(torch.TypedNumberList<double>([1,2,3,4,5,6]),[3,2]);
    print(c);
    torch.Tensor d=torch.eye(3,2);
    print(d);
     var e=torch.sum(d);
     print(e);
     
     var g=torch.div(c,d,inplace: true);
     
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
