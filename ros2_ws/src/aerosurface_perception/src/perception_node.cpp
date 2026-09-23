#include <chrono>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>

#include "aerosurface/runtime.hpp"

namespace aerosurface {
class PerceptionNode final : public rclcpp::Node {
 public:
  PerceptionNode() : Node("aerosurface_perception") {
    width_ = declare_parameter("input_width", 160);
    height_ = declare_parameter("input_height", 128);
    threshold_ = static_cast<float>(declare_parameter("confidence_threshold", .65));
    margin_ = declare_parameter("erosion_margin", 3);
    visualize_ = declare_parameter("visualization", true);
    benchmark_ = declare_parameter("benchmark_mode", false);
    const auto backend = declare_parameter<std::string>("backend", "onnxruntime_cpu");
    const auto precision = declare_parameter<std::string>("precision", "fp32");
    const auto model = declare_parameter<std::string>("model_path", "");
    const auto topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const auto output = declare_parameter<std::string>("output_prefix", "/aerosurface");
    const int threads = declare_parameter("threads", 2);
    max_age_ = declare_parameter("max_image_age_seconds", .5);
    if (backend != "onnxruntime_cpu" || precision != "fp32")
      throw std::invalid_argument(
          "This C++ build supports onnxruntime_cpu/fp32; no silent fallback");
    if (!std::isfinite(threshold_) || threshold_ < 0 || threshold_ > 1 || margin_ < 0 ||
        margin_ > 64 || !std::isfinite(max_age_) || max_age_ <= 0)
      throw std::invalid_argument("Invalid confidence, margin or maximum image age");
    runtime_ = std::make_unique<Runtime>(model, width_, height_, threads);
    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    mask_ = create_publisher<sensor_msgs::msg::Image>(output + "/mask", qos);
    roi_ = create_publisher<sensor_msgs::msg::Image>(output + "/sandable", qos);
    avoid_ = create_publisher<sensor_msgs::msg::Image>(output + "/avoid", qos);
    defect_ = create_publisher<sensor_msgs::msg::Image>(output + "/defect", qos);
    protected_ = create_publisher<sensor_msgs::msg::Image>(output + "/protected", qos);
    overlay_ = create_publisher<sensor_msgs::msg::Image>(output + "/overlay", qos);
    diagnostics_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    sub_ = create_subscription<sensor_msgs::msg::Image>(
        topic, qos, [this](sensor_msgs::msg::Image::ConstSharedPtr message) { process(*message); });
    watchdog_ = create_wall_timer(std::chrono::milliseconds(100), [this] {
      if (!stale_reported_ &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - last_input_).count() >
              max_age_) {
        stale_reported_ = true;
        auto header = last_header_;
        header.stamp = now();
        const std::vector<std::uint8_t> unknown(pixels(width_, height_), 0),
            all_avoid(unknown.size(), 255);
        publish(mask_, header, unknown);
        publish(roi_, header, unknown);
        publish(avoid_, header, all_avoid);
        publish(defect_, header, unknown);
        publish(protected_, header, unknown);
        diagnostic(header, false, "Camera timeout: candidate ROI cleared", 0., max_age_);
      }
    });
    RCLCPP_INFO(get_logger(), "Ready: %dx%d ONNX Runtime CPU FP32, latest image QoS", width_,
                height_);
  }

 private:
  using Publisher = rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr;
  void publish(const Publisher& publisher, const std_msgs::msg::Header& header,
               const std::vector<std::uint8_t>& values, bool color = false) {
    sensor_msgs::msg::Image message;
    message.header = header;
    message.width = static_cast<uint32_t>(width_);
    message.height = static_cast<uint32_t>(height_);
    message.encoding = color ? "rgb8" : "mono8";
    message.step = message.width * (color ? 3 : 1);
    message.is_bigendian = false;
    message.data = values;
    publisher->publish(message);
  }
  void diagnostic(const std_msgs::msg::Header& header, bool ok, const std::string& text,
                  double latency, double age) {
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header = header;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "aerosurface/perception";
    status.hardware_id = "onnxruntime_cpu";
    status.level = ok ? diagnostic_msgs::msg::DiagnosticStatus::OK
                      : diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = text;
    for (const auto& entry : std::vector<std::pair<std::string, std::string>>{
             {"callback_ms", std::to_string(latency)},
             {"input_age_seconds", std::to_string(age)},
             {"processed", std::to_string(processed_)},
             {"errors", std::to_string(errors_)}}) {
      diagnostic_msgs::msg::KeyValue value;
      value.key = entry.first;
      value.value = entry.second;
      status.values.push_back(value);
    }
    array.status.push_back(status);
    diagnostics_->publish(array);
  }
  void process(const sensor_msgs::msg::Image& message) {
    const auto start = std::chrono::steady_clock::now();
    const auto elapsed = [&] {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
          .count();
    };
    double age = 0.;
    last_input_ = start;
    last_header_ = message.header;
    stale_reported_ = false;
    try {
      age = (now() - rclcpp::Time(message.header.stamp, get_clock()->get_clock_type())).seconds();
      if (age > max_age_ || age < -.05) throw std::runtime_error("Stale image or clock mismatch");
      if (message.width > 8192 || message.height > 8192 || !message.width || !message.height ||
          (message.encoding != "rgb8" && message.encoding != "bgr8") ||
          message.step < message.width * 3 ||
          message.data.size() < static_cast<std::size_t>(message.step) * message.height)
        throw std::invalid_argument("Invalid image buffer/encoding; expected rgb8 or bgr8");
      Image image{
          static_cast<int>(message.width), static_cast<int>(message.height),
          std::vector<std::uint8_t>(static_cast<std::size_t>(message.width) * message.height * 3)};
      for (uint32_t y = 0; y < message.height; ++y)
        for (uint32_t x = 0; x < message.width; ++x)
          for (int c = 0; c < 3; ++c)
            image.rgb[(static_cast<std::size_t>(y) * message.width + x) * 3 + c] =
                message.data[static_cast<std::size_t>(y) * message.step + x * 3 +
                             (message.encoding == "bgr8" ? 2 - c : c)];
      const auto regions =
          postprocess(runtime_->infer(image), width_, height_, threshold_, margin_);
      publish(mask_, message.header, regions.mask);
      publish(roi_, message.header, regions.sandable);
      publish(avoid_, message.header, regions.avoid);
      publish(defect_, message.header, regions.defect);
      publish(protected_, message.header, regions.protected_region);
      if (visualize_) {
        const auto input = preprocess(image, width_, height_);
        Image resized{width_, height_, std::vector<std::uint8_t>(input.size())};
        const auto n = pixels(width_, height_);
        for (std::size_t i = 0; i < n; ++i)
          for (int c = 0; c < 3; ++c)
            resized.rgb[i * 3 + c] = static_cast<std::uint8_t>(std::round(input[c * n + i] * 255));
        publish(overlay_, message.header, overlay(resized, regions), true);
      }
      ++processed_;
      diagnostic(message.header, true, "Image-space candidates only", elapsed(), age);
      if (benchmark_)
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "callback %.3f ms, input age %.3f s",
                             elapsed(), age);
    } catch (const std::exception& error) {
      ++errors_;
      const std::vector<std::uint8_t> unknown(pixels(width_, height_), 0),
          all_avoid(unknown.size(), 255);
      publish(mask_, message.header, unknown);
      publish(roi_, message.header, unknown);
      publish(avoid_, message.header, all_avoid);
      publish(defect_, message.header, unknown);
      publish(protected_, message.header, unknown);
      diagnostic(message.header, false, error.what(), elapsed(), age);
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s", error.what());
    }
  }
  int width_{}, height_{}, margin_{};
  float threshold_{};
  bool visualize_{}, benchmark_{};
  double max_age_{};
  uint64_t processed_{}, errors_{};
  std::unique_ptr<Runtime> runtime_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  std::chrono::steady_clock::time_point last_input_{std::chrono::steady_clock::now()};
  std_msgs::msg::Header last_header_;
  bool stale_reported_{false};
  Publisher mask_, roi_, avoid_, defect_, protected_, overlay_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_;
};
}  // namespace aerosurface

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    rclcpp::spin(std::make_shared<aerosurface::PerceptionNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("aerosurface"), "Startup failed: %s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
