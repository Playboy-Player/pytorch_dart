#include "include/pytorch_dart/pytorch_dart_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "pytorch_dart_plugin.h"

void PytorchDartPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  pytorch_dart::PytorchDartPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
