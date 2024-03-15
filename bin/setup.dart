import 'dart:io';
Future<void> runScript() async {
  
  // 将相对路径转换为绝对路径
  var workingDirectory = Directory(Directory.current.path).absolute.path;
  print('Current working directory: ${Directory.current.path}');
  var result = await Process.run(
    'sh',
    ['./scripts/init_linux.sh'], // 这里的路径是相对于当前工作目录的
    // 你可以设置脚本的工作目录
    workingDirectory: workingDirectory,
  );
  print(result.stdout);
  print(result.stderr);
  if (result.exitCode != 0) {
  print('Script failed: exit code ${result.exitCode}');
}
}

void main(List<String> arguments) async{
  print('Setting up pytorch_dart...');
  await runScript();
}