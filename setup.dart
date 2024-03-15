import 'dart:io';
Future<void> runScript() async {
  var result = await Process.run(
    'dart',
    ['./scripts/init_linux.sh'], // 这里的路径是相对于当前工作目录的
    // 你可以设置脚本的工作目录
  );
  print(result.stdout);
  print(result.stderr);
}

void main(List<String> arguments) {
  print('Setting up pytorch_dart...');
  
}