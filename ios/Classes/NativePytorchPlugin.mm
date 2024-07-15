#import "NativePytorchPlugin.h"
#if __has_include(<native_pytorch/native_pytorch-Swift.h>)
#import <native_pytorch/native_pytorch-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "native_pytorch-Swift.h"
#endif

@implementation NativePytorchPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftNativePytorchPlugin registerWithRegistrar:registrar];
}
@end
