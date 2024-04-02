//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <pytorch_dart/pytorch_dart_plugin.h>

void fl_register_plugins(FlPluginRegistry* registry) {
  g_autoptr(FlPluginRegistrar) pytorch_dart_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "PytorchDartPlugin");
  pytorch_dart_plugin_register_with_registrar(pytorch_dart_registrar);
}
