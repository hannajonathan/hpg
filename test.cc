#include "hpg.hpp"

#include <complex>
#include <iostream>
#include <random>
#include <vector>

struct MyCF2 final
  : public hpg::CF2 {

  std::vector<std::complex<float>> values;

  MyCF2() {}

  MyCF2(
    unsigned oversampling_,
    const std::array<unsigned, 2>& size,
    const std::vector<std::complex<float>>& values_)
    : values(values_) {

    oversampling = oversampling_;
    extent[0] = size[0] * oversampling;
    extent[1] = size[1] * oversampling;
  }

  std::complex<float>
  operator()(unsigned x, unsigned y) const override {
    return values[x * extent[1] + y];
  }
};

int
main(int argc, char* argv[]) {

  MyCF2 cf2;
  {
    const unsigned oversampling = 10;
    const std::array<unsigned, 2> size{31, 25};
    const unsigned num_values = oversampling * size[0] * oversampling * size[1];
    std::vector<std::complex<float>> values;
    values.reserve(num_values);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto i = 0; i < num_values; ++i)
      values.emplace_back(dist(gen), dist(gen));
    cf2 = MyCF2(oversampling, size, values);
  }

  hpg::initialize();
#ifdef HPG_ENABLE_SERIAL
  std::cout << "Serial" << std::endl;
  {
    auto st0 =
      hpg::GridderState(hpg::Device::Serial, {1000, 2000, 3}, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 =
      std::move(st2).set_convolution_function(hpg::Device::Serial, cf2);
  }
  {
    auto g0 = hpg::Gridder(hpg::Device::Serial, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
    g0.set_convolution_function(hpg::Device::OpenMP, cf2);
  }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
  std::cout << "Cuda" << std::endl;
  {
    auto st0 =
      hpg::GridderState(hpg::Device::Cuda, {1000, 2000, 3}, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 =
      std::move(st2).set_convolution_function(hpg::Device::OpenMP, cf2);
  }
  {
    auto g0 = hpg::Gridder(hpg::Device::Cuda, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
    g0.set_convolution_function(hpg::Device::OpenMP, cf2);
  }
#endif // HPG_ENABLE_CUDA
  hpg::finalize();
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
