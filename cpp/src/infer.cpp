#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "aerosurface/runtime.hpp"

namespace {
aerosurface::Image read_ppm(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int w = 0, h = 0, max = 0;
  if (!(file >> magic >> w >> h >> max) || magic != "P6" || max != 255)
    throw std::runtime_error("Expected binary P6 PPM, maxval 255, no comments");
  file.get();
  aerosurface::Image result{w, h, std::vector<std::uint8_t>(aerosurface::pixels(w, h) * 3)};
  if (!file.read(reinterpret_cast<char*>(result.rgb.data()),
                 static_cast<std::streamsize>(result.rgb.size())))
    throw std::runtime_error("Truncated PPM");
  return result;
}
void save(const std::string& path, const std::vector<std::uint8_t>& data, int w, int h,
          bool color) {
  std::ofstream stream(path, std::ios::binary);
  stream << (color ? "P6\n" : "P5\n") << w << ' ' << h << "\n255\n";
  stream.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
  if (!stream) throw std::runtime_error("Cannot save output");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 6)
      throw std::invalid_argument(
          "Usage: aerosurface_infer model.onnx input.ppm output_prefix width height");
    const int w = std::stoi(argv[4]), h = std::stoi(argv[5]);
    aerosurface::Runtime runtime(argv[1], w, h);
    const auto image = read_ppm(argv[2]);
    for (int i = 0; i < 20; ++i) runtime.infer(image);
    std::vector<double> times;
    aerosurface::Regions regions;
    for (int i = 0; i < 100; ++i) {
      const auto start = std::chrono::steady_clock::now();
      regions = aerosurface::postprocess(runtime.infer(image), w, h);
      times.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count());
    }
    const std::string prefix = argv[3];
    save(prefix + "_mask.pgm", regions.mask, w, h, false);
    save(prefix + "_sandable.pgm", regions.sandable, w, h, false);
    save(prefix + "_avoid.pgm", regions.avoid, w, h, false);
    save(prefix + "_defect.pgm", regions.defect, w, h, false);
    save(prefix + "_protected.pgm", regions.protected_region, w, h, false);
    auto input = aerosurface::preprocess(image, w, h);
    aerosurface::Image resized{w, h, std::vector<std::uint8_t>(input.size())};
    const auto n = aerosurface::pixels(w, h);
    for (std::size_t i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c)
        resized.rgb[i * 3 + c] = static_cast<std::uint8_t>(std::round(input[c * n + i] * 255));
    save(prefix + "_overlay.ppm", aerosurface::overlay(resized, regions), w, h, true);
    std::sort(times.begin(), times.end());
    double total = 0;
    for (const auto value : times) total += value;
    const std::string json =
        "{\"backend\":\"cpp_ort_cpu\",\"scope\":\"preprocess_infer_postprocess\",\"iterations\":"
        "100,\"warmup\":20,\"mean_ms\":" +
        std::to_string(total / times.size()) + ",\"p50_ms\":" + std::to_string(times[49]) +
        ",\"p95_ms\":" + std::to_string(times[94]) + ",\"fps\":" + std::to_string(100000 / total) +
        "}";
    std::ofstream(prefix + "_benchmark.json") << json;
    std::cout << json << '\n';
  } catch (const std::exception& error) {
    std::cerr << "AeroSurface error: " << error.what() << '\n';
    return 1;
  }
}
