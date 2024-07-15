#import "PytorchDartPlugin.h"
#if __has_include(<pytorch_dart/pytorch_dart-Swift.h>)
#import <pytorch_dart/pytorch_dart-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "pytorch_dart-Swift.h"
#endif

@implementation PytorchDartPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftPytorchDartPlugin registerWithRegistrar:registrar];
}
@end
