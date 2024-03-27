import 'dart:io';
import 'package:package_config/package_config.dart';
import 'package:path/path.dart' as path;
import 'package:collection/collection.dart';

Future<void> runScript4Linux() async {
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
      
      var packageRootPath = path.fromUri(package.packageUriRoot);
      print('plugin root path:$packageRootPath');
      
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
    ['../scripts/init_linux.sh'], // 这里的路径是相对于当前工作目录的
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

Future<void> runScript4AndoridOnLinux() async {
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
      
      var packageRootPath = path.fromUri(package.packageUriRoot);
      print('plugin root path:$packageRootPath');
      
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
    ['../scripts/init_android_linux.sh'], // 这里的路径是相对于当前工作目录的
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
  String command=arguments[0];
  if(command == '--platform')
  {
    if(arguments[1]=='linux'){
  if(Platform.isLinux){
  await runScript4Linux();
  }
    }
  if(arguments[1]=='android')
  {if(Platform.isLinux){
    await runScript4AndoridOnLinux();
  }
  }
  
  }

}