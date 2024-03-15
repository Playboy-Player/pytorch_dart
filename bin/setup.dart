import 'dart:io';
import 'package:package_config/package_config.dart';
import 'package:path/path.dart' as path;
import 'package:collection/collection.dart';

Future<void> runScript() async {
 String? currentDirectory;
  
 // 加载当前包的配置信息
  var packageConfig = await findPackageConfig(Directory.current);
  if (packageConfig != null) {
    // 查找指定插件的Package对象
    var packageName = 'pytorch_dart'; 
    var package = packageConfig.packages.firstWhereOrNull(
      (pkg) => pkg.name == packageName,
    );
    if (package != null) {
      // 转换Uri为文件路径
      var packageRootPath = path.fromUri(package.packageUriRoot);
      print('plugin root path:$packageRootPath');
      // 在这里使用 packageRootPath，确保在它的作用域内
      currentDirectory = packageRootPath;
      print('currect directory$currentDirectory');
    } else {
      print('cant find $packageName');
    }
  } else {
    print('cant find packageConfig');
  }
  
  
  ProcessStartMode processStartMode = ProcessStartMode.inheritStdio;
  ProcessStartMode executable = ProcessStartMode.detached;
  
  
  print(currentDirectory);
  var result = await Process.start(
    'sh',
    ['../scripts/init_linux_and_android.sh'], // 这里的路径是相对于当前工作目录的
    // 设置脚本的工作目录
    workingDirectory: currentDirectory!,
    mode: processStartMode,
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