#pragma once
#include <onnxruntime_cxx_api.h>

#include <filesystem>

#include "aerosurface/core.hpp"

namespace aerosurface {
class Runtime {
 public:
  Runtime(const std::filesystem::path& model, int width, int height, int threads = 2);
  std::vector<float> infer(const Image& image);

 private:
  int width_, height_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "aerosurface"};
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
};
}  // namespace aerosurface
