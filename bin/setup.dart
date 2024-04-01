import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:archive/archive.dart';
import 'package:package_config/package_config.dart';
import 'package:path/path.dart' as path;
import 'package:collection/collection.dart';

Future<void> setup4Linux() async {
var url = Uri.parse('https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip');
  var filename = 'libtorch-shared-with-deps-latest.zip';

  // 下载文件
  var response = await http.get(url);
  if (response.statusCode == 200) {
    var file = File(filename);
    await file.writeAsBytes(response.bodyBytes);
    print('File downloaded and saved as $filename');

    // 读取ZIP文件
    var bytes = await file.readAsBytes();
    var archive = ZipDecoder().decodeBytes(bytes);

    // 创建新的目录来解压缩文件
    var newFolder = 'libtorch-linux';
    var newPath = path.join(Directory.current.path, newFolder);
    Directory(newPath).createSync(recursive: true);

    // 从压缩包中提取文件
    for (var file in archive) {
      var fileName = file.name;

      // 如果是根目录下的文件夹，则跳过不解压缩
      if (file.isFile || path.split(fileName).length != 1) {
        var data = file.content as List<int>;
        var outputPath = path.join(newPath, fileName);

        // 确保父目录存在
        Directory(path.dirname(outputPath)).createSync(recursive: true);

        if (file.isFile) {
          File(outputPath)..writeAsBytesSync(data);
        } else {
          // 如果项是文件夹，则创建文件夹
          Directory(outputPath).createSync(recursive: true);
        }
        print('Extracted: $outputPath');
      }
    }
    print('Files extracted to $newFolder');

    // 删除下载的ZIP文件
    await file.delete();
    print('ZIP file deleted');
  } else {
    print('Failed to download file: ${response.statusCode}');
  }
}

Future<void> setup4Windows() async {
var url = Uri.parse('https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.2.2%2Bcpu.zip');
  var filename = 'libtorch-win-shared-with-deps-2.2.2+cpu.zip';

  // 下载文件
  var response = await http.get(url);
  if (response.statusCode == 200) {
    var file = File(filename);
    await file.writeAsBytes(response.bodyBytes);
    print('File downloaded and saved as $filename');

    // 读取ZIP文件
    var bytes = await file.readAsBytes();
    var archive = ZipDecoder().decodeBytes(bytes);

    // 创建新的目录来解压缩文件
    var newFolder = 'libtorch-windows';
    var newPath = path.join(Directory.current.path, newFolder);
    Directory(newPath).createSync(recursive: true);

    // 从压缩包中提取文件
    for (var file in archive) {
      var fileName = file.name;

      // 如果是根目录下的文件夹，则跳过不解压缩
      if (file.isFile || path.split(fileName).length != 1) {
        var data = file.content as List<int>;
        var outputPath = path.join(newPath, fileName);

        // 确保父目录存在
        Directory(path.dirname(outputPath)).createSync(recursive: true);

        if (file.isFile) {
          File(outputPath)..writeAsBytesSync(data);
        } else {
          // 如果项是文件夹，则创建文件夹
          Directory(outputPath).createSync(recursive: true);
        }
        print('Extracted: $outputPath');
      }
    }
    print('Files extracted to $newFolder');

    // 删除下载的ZIP文件
    await file.delete();
    print('ZIP file deleted');
  } else {
    print('Failed to download file: ${response.statusCode}');
  }
}



Future<void> setup4Android() async {
  // 文件的URL
  var url = Uri.parse(
      'https://oss.sonatype.org/service/local/artifact/maven/redirect?r=releases&g=org.pytorch&a=pytorch_android&v=2.1.0&e=aar');

  var response = await http.get(url);
  if (response.statusCode == 200) {
    var archive = ZipDecoder().decodeBytes(response.bodyBytes);

    // 存储旧文件夹名到新文件夹名的映射关系
    Map<String, String> dirRenames = {
      'jni': 'lib',
      'headers': 'include',
    };

    // 循环每个文件
    for (var file in archive) {
      // 获得文件路径的所有组成部分
      var filePathParts = path.split(file.name);

      // 使用映射来更新文件路径部分
      var updatedFilePathParts = filePathParts.map((part) {
        return dirRenames[part] ?? part; // 如果映射中存在则替换，否则使用原始部分
      }).toList();

      // 重新组合文件路径
      var filePath = path.joinAll(updatedFilePathParts);

      if (file.isFile) {
        final data = file.content as List<int>;
        final outputPath =
            path.join("${Directory.current.path}/libtorch-android", filePath);
        // 确保父目录存在
        final directory = Directory(path.dirname(outputPath));
        if (!directory.existsSync()) {
          directory.createSync(recursive: true);
        }
        File(outputPath)..writeAsBytesSync(data);
      } else {
        Directory(
            path.join("${Directory.current.path}/libtorch-android", filePath))
          ..createSync(recursive: true);
      }
    }
    print('Extraction done.');
  } else {
    print('Failed to download file: ${response.statusCode}');
  }
}

void main(List<String> arguments) async {
  print('Setting up pytorch_dart...');
  String command = arguments[0];
  if (command == '--platform') {
    if (arguments[1] == 'linux') {
      await setup4Linux();
    } else if (arguments[1] == 'windows') {
      await setup4Windows();
    }
    if (arguments[1] == 'android') {
      await setup4Android();
    }
  }
}
