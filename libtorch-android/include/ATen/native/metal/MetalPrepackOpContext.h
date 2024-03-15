#pragma once

#include <ATen/Tensor.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace metal {

using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    c10::optional<Scalar>,
    c10::optional<Scalar>>;

class Conv2dOpContext : public torch::jit::CustomClassHolder {
 public:
  SerializationTypeConv2dPrePack pack() {
    return std::make_tuple(
        weight_,
        bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }
  Conv2dOpContext() = delete;
  Conv2dOpContext(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation,
      int64_t groups,
      const c10::optional<Scalar>& output_min,
      const c10::optional<Scalar>& output_max)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        stride_(stride),
        padding_(padding),
        dilation_(dilation),
        groups_(groups),
        output_min_(output_min),
        output_max_(output_max) {}

  ~Conv2dOpContext() {
    if (releaseCallback_) {
      releaseCallback_(conv2dOp_);
    }
  }

  void release_resources() override {
    if (releaseCallback_) {
      releaseCallback_(conv2dOp_);
    }
  }

  const Tensor& get_weight() const {
    return weight_;
  }

  const c10::optional<Tensor>& get_bias() const {
    return bias_;
  }

  const std::vector<int64_t>& get_stride() const {
    return stride_;
  }

  const std::vector<int64_t>& get_padding() const {
    return padding_;
  }

  const std::vector<int64_t>& get_dilation() const {
    return dilation_;
  }

  int64_t get_groups() const {
    return groups_;
  }

  const c10::optional<Scalar>& get_output_min() const {
    return output_min_;
  }

  const c10::optional<Scalar>& get_output_max() const {
    return output_max_;
  }

  void set_conv2dOpPtr(void* ptr) {
      conv2dOp_ = ptr;
  }

  void* get_conv2dOpPtr() const {
    return conv2dOp_;
  }

  void set_releaseCallback(const std::function<void(void*)>& func) {
    releaseCallback_ = func;
  }

  std::function<void(void*)>& get_releaseCallback() {
     return releaseCallback_;
  }

  private:
    Tensor weight_;
    c10::optional<Tensor> bias_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> dilation_;
    int64_t groups_;
    c10::optional<Scalar> output_min_;
    c10::optional<Scalar> output_max_;
    std::function<void(void*)> releaseCallback_ = nullptr;
    void* conv2dOp_ = nullptr; // reserved to hold MPSCNNConv2dOp objects
};

using SerializationTypeLinearPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    c10::optional<Scalar>,
    c10::optional<Scalar>>;

class LinearOpContext : public torch::jit::CustomClassHolder {
 public:
  SerializationTypeLinearPrePack pack() {
    return std::make_tuple(weight_, bias_, output_min_, output_max_);
  }
  LinearOpContext() = delete;
  LinearOpContext(
      at::Tensor&& weight,
      c10::optional<at::Tensor>&& bias,
      const c10::optional<Scalar>& output_min,
      const c10::optional<Scalar>& output_max)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        output_min_(output_min),
        output_max_(output_max) {}

  ~LinearOpContext() {
    if (releaseCallback_) {
      releaseCallback_(opaqueOpPtr_);
    }
  }

  void release_resources() override {
    if (releaseCallback_) {
      releaseCallback_(opaqueOpPtr_);
    }
  }

  const Tensor& get_weight() const {
    return weight_;
  }

  const c10::optional<Tensor>& get_bias() const {
    return bias_;
  }

  const c10::optional<Scalar>& get_output_min() const {
    return output_min_;
  }

  const c10::optional<Scalar>& get_output_max() const {
    return output_max_;
  }

  void set_opaqueOpPtr(void* ptr) {
    opaqueOpPtr_ = ptr;
  }

  void* get_opaqueOpPtr() const {
    return opaqueOpPtr_;
  }

  void set_releaseCallback(const std::function<void(void*)>& func) {
    releaseCallback_ = func;
  }

  std::function<void(void*)>& get_releaseCallback() {
    return releaseCallback_;
  }

 private:
  Tensor weight_;
  c10::optional<Tensor> bias_;
  c10::optional<Scalar> output_min_;
  c10::optional<Scalar> output_max_;
  void* opaqueOpPtr_ = nullptr; // reserved to hold MPSCNNFullyConnected objects
  std::function<void(void*)> releaseCallback_ = nullptr;
};

} // namespace metal
} // namespace native
} // namespace at
