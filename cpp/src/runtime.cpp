#include "aerosurface/runtime.hpp"

#include <array>
#include <string>

namespace aerosurface {
Runtime::Runtime(const std::filesystem::path& model, int width, int height, int threads)
    : width_(width), height_(height) {
  pixels(width, height);
  if (threads < 1) throw std::invalid_argument("threads must be positive");
  options_.SetIntraOpNumThreads(threads);
  options_.SetInterOpNumThreads(1);
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = Ort::Session(env_, model.c_str(), options_);
  if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
    throw std::runtime_error("Expected one model input/output");
  Ort::AllocatorWithDefaultOptions allocator;
  if (std::string(session_.GetInputNameAllocated(0, allocator).get()) != "rgb" ||
      std::string(session_.GetOutputNameAllocated(0, allocator).get()) != "logits")
    throw std::runtime_error("Model names must be rgb/logits");
  const auto input_info = session_.GetInputTypeInfo(0);
  const auto output_info = session_.GetOutputTypeInfo(0);
  const auto input = input_info.GetTensorTypeAndShapeInfo();
  const auto output = output_info.GetTensorTypeAndShapeInfo();
  if (input.GetShape() != std::vector<int64_t>{1, 3, height, width} ||
      output.GetShape() != std::vector<int64_t>{1, 5, height, width} ||
      input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("Model must expose static float32 [1,3,H,W] -> [1,5,H,W]");
}

std::vector<float> Runtime::infer(const Image& image) {
  auto data = preprocess(image, width_, height_);
  const std::array<int64_t, 4> shape{1, 3, height_, width_};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor =
      Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
  constexpr std::array<const char*, 1> inputs{"rgb"}, outputs{"logits"};
  auto result =
      session_.Run(Ort::RunOptions{nullptr}, inputs.data(), &tensor, 1, outputs.data(), 1);
  const auto* values = result[0].GetTensorData<float>();
  return {values, values + pixels(width_, height_) * 5};
}
}  // namespace aerosurface
