#include "hpg.hpp"

#include <complex>
#include <iostream>
#include <random>
#include <vector>

struct MyCF2
  : public hpg::CF2 {

  std::vector<std::complex<float>> values;

  MyCF2(
    unsigned oversampling_,
    unsigned size,
    const std::vector<std::complex<float>>& values_)
    : values(values_) {

    oversampling = oversampling_;
    extent[0] = size * oversampling;
    extent[1] = size * oversampling;
  }

  std::complex<float>
  operator()(unsigned x, unsigned y) const override {
    return values[x * oversampling + y];
  }
};

int
main(int argc, char* argv[]) {

  const unsigned cf2_oversampling = 10;
  const unsigned cf2_size = 31;
  const unsigned cf2_num_values =
    cf2_oversampling * cf2_size * cf2_oversampling * cf2_size;
  std::vector<std::complex<float>> cf2_values;
  cf2_values.reserve(cf2_num_values);
  {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto i = 0; i < cf2_num_values; ++i)
      cf2_values.emplace_back(dist(gen), dist(gen));
  }
  MyCF2 cf2(cf2_oversampling, cf2_size, cf2_values);

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
  }
  {
    auto g0 = hpg::Gridder(hpg::Device::Cuda, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
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
