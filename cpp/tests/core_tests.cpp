#include <iostream>
#include <limits>

#include "aerosurface/core.hpp"

void require(bool value) {
  if (!value) throw std::runtime_error("Contract check failed");
}
template <class F>
void throws(F&& fn) {
  bool caught = false;
  try {
    fn();
  } catch (const std::exception&) {
    caught = true;
  }
  require(caught);
}
int main() {
  aerosurface::Image image{2, 1, {255, 0, 0, 0, 255, 0}};
  const auto x = aerosurface::preprocess(image, 4, 2);
  require(x.size() == 24 && x[0] == 1 && x[2] == 0 && x[10] == 1);
  throws([&] { aerosurface::preprocess(image, 0, 2); });
  throws([&] { aerosurface::preprocess({2, 2, {}}, 2, 2); });
  std::vector<float> logits(5 * 121, 0);
  require(aerosurface::postprocess(logits, 11, 11).sandable[60] == 0);
  std::fill(logits.begin() + 121, logits.begin() + 242, 10.F);
  logits[2 * 121 + 60] = 20;
  const auto r = aerosurface::postprocess(logits, 11, 11, .65F, 2);
  require(r.sandable[2 * 11 + 2] == 255 && r.sandable[60] == 0 && r.sandable[0] == 0);
  require(r.protected_region[60] == 255 && r.avoid[60] == 255);
  logits[0] = std::numeric_limits<float>::quiet_NaN();
  throws([&] { aerosurface::postprocess(logits, 11, 11); });
  throws([&] { aerosurface::postprocess({}, 11, 11); });
  std::cout << "C++ preprocessing, rejection, confidence and ROI contracts passed\n";
}
