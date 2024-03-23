// Copyright 2020, GoTorch Authors
#include "torch.h"
#if defined(__GNUC__)
// Attributes to prevent 'unused' function from being removed and to make it visible
#define FUNCTION_ATTRIBUTE __attribute__((visibility("default"))) __attribute__((used))
#elif defined(_MSC_VER)
// Marking a function for export
#define FUNCTION_ATTRIBUTE __declspec(dllexport)
#endif
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

FUNCTION_ATTRIBUTE const char *exception_str(const char *e) {
  auto len = strlen(e);
  auto r = new char[len + 1];
  snprintf(r, len + 1, "%s", e);
  return r;
}

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations
////////////////////////////////////////////////////////////////////////////////

FUNCTION_ATTRIBUTE const char *RandN(int64_t *size, int64_t length, int64_t requires_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::randn(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Rand(int64_t *size, int64_t length, int64_t requires_grad,
                 Tensor *result) {
  try {
    at::Tensor t =
        torch::rand(torch::IntArrayRef(size, length),
                    at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Stack(Tensor *tensors, int64_t tensors_size, int64_t dim,
                  Tensor *result) {
  try {
    std::vector<torch::Tensor> data;
    while (data.size() < tensors_size) data.push_back(**tensors++);
    auto out = at::stack(data, dim);
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Empty(int64_t *size, int64_t length, int64_t requires_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::empty(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.ones
FUNCTION_ATTRIBUTE const char *Ones(int64_t *size, int64_t length, int64_t requires_grad,
                 Tensor *result) {
  try {
    at::Tensor t =
        torch::ones(torch::IntArrayRef(size, length),
                    at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.eye
FUNCTION_ATTRIBUTE const char *Eye(int64_t n, int64_t m, int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t =
        torch::eye(n, m, at::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.full, only for float32
FUNCTION_ATTRIBUTE const char *Full(int64_t *size, int64_t length, float value,
                 int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t =
        torch::full(torch::IntArrayRef(size, length), value,
                    torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.arange
FUNCTION_ATTRIBUTE const char *Arange(float start, float end, float step, int64_t requires_grad,
                   Tensor *result) {
  try {
    at::Tensor t = torch::arange(
        start, end, step, torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.linspace
FUNCTION_ATTRIBUTE const char *Linspace(float start, float end, int64_t steps,
                     int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t = torch::linspace(
        start, end, steps, torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// torch.logspace
FUNCTION_ATTRIBUTE const char *Logspace(float start, float end, int64_t steps, double base,
                     int64_t requires_grad, Tensor *result) {
  try {
    at::Tensor t =
        torch::logspace(start, end, steps, base,
                        torch::TensorOptions().requires_grad(requires_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Equal(Tensor a, Tensor b, int64_t *result) {
  try {
    *result = at::equal(*a, *b) ? 1 : 0;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c = at::mm(*a, *b);
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Sum(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *SumByDim(Tensor a, int64_t dim, int8_t keepDim, Tensor *result) {
  try {
    *result = new at::Tensor(a->sum(dim, keepDim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Relu(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->relu());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result) {
  try {
    *result = new at::Tensor(at::leaky_relu(*a, negative_slope));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Tanh(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->tanh());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Sigmoid(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->sigmoid());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Add(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(torch::add(*a, *other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Add_(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(a->add_(*other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Sub(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(torch::sub(*a, *other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Sub_(Tensor a, Tensor other, float alpha, Tensor *result) {
  try {
    *result = new at::Tensor(a->sub_(*other, alpha));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Mul(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::mul(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Mul_(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->mul_(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Div(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::div(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Div_(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->div_(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Permute(Tensor a, int64_t *dims, int64_t dims_size,
                    Tensor *result) {
  try {
    c10::ArrayRef<int64_t> d(dims, dims_size);
    *result = new at::Tensor(a->permute(d));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *AllClose(Tensor a, Tensor b, int64_t *result) {
  try {
    *result = at::allclose(*a, *b) ? 1 : 0;
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Flatten(Tensor a, int64_t startDim, int64_t endDim,
                    Tensor *result) {
  try {
    *result = new at::Tensor(torch::flatten(*a, startDim, endDim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *TopK(Tensor a, int64_t k, int64_t dim, int8_t largest,
                 int8_t sorted, Tensor *values, Tensor *indices) {
  try {
    auto outputs = torch::topk(*a, k, dim, largest, sorted);
    *values = new at::Tensor(std::get<0>(outputs));
    *indices = new at::Tensor(std::get<1>(outputs));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor *result) {
  try {
    *result = new at::Tensor(torch::transpose(*a, dim0, dim1));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *ExpandAs(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(a->expand_as(*other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Eq(Tensor a, Tensor other, Tensor *result) {
  try {
    *result = new at::Tensor(torch::eq(*a, *other));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor *result) {
  try {
    *result = new at::Tensor(torch::index_select(*a, dim, *index));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len) {
  try {
    *result = new at::Tensor(a->view(torch::IntArrayRef(size, size_len)));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->log_softmax(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Squeeze(Tensor a, Tensor *result) {
  try {
    *result = new at::Tensor(a->squeeze());
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *SqueezeWithDim(Tensor a, int64_t dim, Tensor *result) {
  try {
    *result = new at::Tensor(a->squeeze(dim));
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

// We use the pointer int64_t* to represent an optional int64_t parameter -- the
// value nullptr indicate not-specified.  Please be aware that we need only one
// "pointerized" parameter because C++ doesn't allow named parameters and the
// rest optional parameters don't need to be pointerized.
FUNCTION_ATTRIBUTE const char *Argmin(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result) {
  try {
    if (dim == nullptr) {
      *result = new at::Tensor(a->argmin());
    } else {
      *result = new at::Tensor(a->argmin(*dim, static_cast<bool>(keepdim)));
    }
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}

FUNCTION_ATTRIBUTE const char *Argmax(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result) {
  try {
    if (dim == nullptr) {
      *result = new at::Tensor(a->argmax());
    } else {
      *result = new at::Tensor(a->argmax(*dim, static_cast<bool>(keepdim)));
    }
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
