#ifndef FLUTTER_PLUGIN_PYTORCH_DART_PLUGIN_H_
#define FLUTTER_PLUGIN_PYTORCH_DART_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace pytorch_dart {

class PytorchDartPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  PytorchDartPlugin();

  virtual ~PytorchDartPlugin();

  // Disallow copy and assign.
  PytorchDartPlugin(const PytorchDartPlugin&) = delete;
  PytorchDartPlugin& operator=(const PytorchDartPlugin&) = delete;

  // Called when a method is called on this plugin's channel from Dart.
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace pytorch_dart

#endif  // FLUTTER_PLUGIN_PYTORCH_DART_PLUGIN_H_
