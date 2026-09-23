#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aerosurface {
struct Image {
  int width{}, height{};
  std::vector<std::uint8_t> rgb;
};

inline std::size_t pixels(int width, int height) {
  if (width <= 0 || height <= 0 || width > 8192 || height > 8192)
    throw std::invalid_argument("Image dimensions must be within 1..8192");
  return static_cast<std::size_t>(width) * height;
}

inline std::vector<float> preprocess(const Image& image, int width, int height) {
  const auto n = pixels(width, height);
  if (image.rgb.size() != pixels(image.width, image.height) * 3)
    throw std::invalid_argument("RGB buffer size mismatch");
  std::vector<float> output(3 * n);
  for (int y = 0; y < height; ++y) {
    const int sy = y * image.height / height;
    for (int x = 0; x < width; ++x) {
      const int sx = x * image.width / width;
      const auto src = (static_cast<std::size_t>(sy) * image.width + sx) * 3;
      const auto dst = static_cast<std::size_t>(y) * width + x;
      for (int c = 0; c < 3; ++c) output[c * n + dst] = image.rgb[src + c] / 255.F;
    }
  }
  return output;
}

struct Regions {
  std::vector<std::uint8_t> mask, sandable, avoid, defect, protected_region;
};

inline Regions postprocess(const std::vector<float>& logits, int width, int height,
                           float threshold = .65F, int margin = 3) {
  const auto n = pixels(width, height);
  if (logits.size() != n * 5 || !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
      margin < 0 || margin > 64)
    throw std::invalid_argument("Invalid logits or postprocess parameters");
  Regions r{std::vector<std::uint8_t>(n), std::vector<std::uint8_t>(n),
            std::vector<std::uint8_t>(n, 255), std::vector<std::uint8_t>(n),
            std::vector<std::uint8_t>(n)};
  for (std::size_t i = 0; i < n; ++i) {
    int best = 0;
    for (int c = 0; c < 5; ++c) {
      if (!std::isfinite(logits[c * n + i])) throw std::runtime_error("Nonfinite logits");
      if (logits[c * n + i] > logits[best * n + i]) best = c;
    }
    float sum = 0;
    for (int c = 0; c < 5; ++c) sum += std::exp(logits[c * n + i] - logits[best * n + i]);
    r.mask[i] = 1.F / sum >= threshold ? static_cast<std::uint8_t>(best) : 0;
    r.defect[i] = r.mask[i] == 3 ? 255 : 0;
    r.protected_region[i] = r.mask[i] == 2 ? 255 : 0;
  }
  for (int y = margin; y < height - margin; ++y) {
    for (int x = margin; x < width - margin; ++x) {
      bool candidate = true;
      for (int dy = -margin; dy <= margin && candidate; ++dy)
        for (int dx = -margin; dx <= margin; ++dx)
          if (r.mask[static_cast<std::size_t>(y + dy) * width + x + dx] != 1) {
            candidate = false;
            break;
          }
      const auto i = static_cast<std::size_t>(y) * width + x;
      r.sandable[i] = candidate ? 255 : 0;
      r.avoid[i] = candidate ? 0 : 255;
    }
  }
  return r;
}

inline std::vector<std::uint8_t> overlay(const Image& image, const Regions& r) {
  constexpr std::array<std::array<int, 3>, 5> colors{
      {{24, 30, 45}, {60, 190, 125}, {245, 190, 40}, {245, 65, 85}, {115, 110, 235}}};
  if (r.mask.size() != pixels(image.width, image.height) || image.rgb.size() != r.mask.size() * 3)
    throw std::invalid_argument("Overlay shape mismatch");
  auto result = image.rgb;
  for (std::size_t i = 0; i < r.mask.size(); ++i) {
    if (r.mask[i] > 4) throw std::invalid_argument("Invalid class id");
    for (int c = 0; c < 3; ++c)
      result[i * 3 + c] =
          static_cast<std::uint8_t>(image.rgb[i * 3 + c] * .55 + colors[r.mask[i]][c] * .45);
  }
  return result;
}
}  // namespace aerosurface
