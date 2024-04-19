//Some functions only available for desktop platforms
#include "tensor4desktop.h"

#include <string>
#include <vector>
#include <iostream>
#include <sstream>


const char *Tensor_Save(Tensor tensor, const char *path) {
  try {
    torch::save(*tensor, std::string(path));
    return nullptr;
  } catch (const c10::Error &e) {
    return exception_str(e.what());
  }
}

const char *Tensor_Load(const char *path, Tensor *tensor) {
  try {
    *tensor = new at::Tensor();
    torch::load(**tensor, path);
    return nullptr;
  } catch (const c10::Error &e) {
    return exception_str(e.what());
  }
}