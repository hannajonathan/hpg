#include "hpg.hpp"

#include <cassert>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

struct MyCF2 final
  : public hpg::CF2 {

  std::vector<std::complex<hpg::cf_fp>> values;

  MyCF2() {}

  MyCF2(
    unsigned oversampling_,
    const std::array<unsigned, 2>& size,
    const std::vector<std::complex<hpg::cf_fp>>& values_)
    : values(values_) {

    oversampling = oversampling_;
    extent[0] = size[0] * oversampling;
    extent[1] = size[1] * oversampling;
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y) const override {
    return values[x * extent[1] + y];
  }
};

template <typename Generator>
MyCF2
create_cf2(Generator& gen) {
  const unsigned oversampling = 10;
  const std::array<unsigned, 2> size{31, 25};
  const unsigned num_values = oversampling * size[0] * oversampling * size[1];
  std::vector<std::complex<hpg::cf_fp>> values;
  values.reserve(num_values);
  std::uniform_real_distribution<hpg::cf_fp> dist(-1.0, 1.0);
  for (auto i = 0; i < num_values; ++i)
    values.emplace_back(dist(gen), dist(gen));
  return MyCF2(oversampling, size, values);
}

template <typename Generator>
void
init_visibilities(
  Generator& gen,
  std::vector<std::complex<hpg::visibility_fp>>& visibilities,
  std::vector<hpg::vis_weight_fp>& visibility_weights,
  std::vector<hpg::vis_frequency_fp>& visibility_frequencies,
  std::vector<hpg::vis_phase_fp>& visibility_phase,
  std::vector<hpg::vis_uvw_t>& visibility_coordinates) {

  const unsigned num_visibilities = 1000000;
  visibilities.clear();
  visibilities.reserve(num_visibilities);
  visibility_weights.clear();
  visibility_weights.reserve(num_visibilities);
  visibility_frequencies.clear();
  visibility_frequencies.reserve(num_visibilities);
  visibility_phase.clear();
  visibility_phase.reserve(num_visibilities);
  visibility_coordinates.clear();
  visibility_coordinates.reserve(num_visibilities);

  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  std::uniform_real_distribution<hpg::vis_uvw_fp> dist_uvw(-475.0, 475.0);
  for (auto i = 0; i < num_visibilities; ++i) {
    visibilities.emplace_back(dist_vis(gen), dist_vis(gen));
    visibility_weights.push_back(dist_weight(gen));
    visibility_frequencies.push_back(2.92513197327302e9);
    visibility_phase.emplace_back(0.0);
    visibility_coordinates.push_back(
      hpg::vis_uvw_t({dist_uvw(gen), dist_uvw(gen), 0.0}));
  }
}

int
main(int argc, char* argv[]) {

  std::mt19937 rng(42);

  MyCF2 cf2 = create_cf2(rng);

  std::vector<std::complex<hpg::visibility_fp>> visibilities;
  std::vector<hpg::vis_weight_fp> visibility_weights;
  std::vector<hpg::vis_frequency_fp> visibility_frequencies;
  std::vector<hpg::vis_phase_fp> visibility_phase;
  std::vector<hpg::vis_uvw_t> visibility_coordinates;
  init_visibilities(
    rng,
    visibilities,
    visibility_weights,
    visibility_frequencies,
    visibility_phase,
    visibility_coordinates);

  hpg::initialize();
#ifdef HPG_ENABLE_SERIAL
  std::cout << "Serial" << std::endl;
  {
    auto st0 =
      hpg::GridderState(hpg::Device::Serial, 4, {1000, 2000, 3}, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 =
      std::move(st2).set_convolution_function(hpg::Device::Serial, cf2);
    assert(st3.max_async_tasks() == 0);
  }
  {
    auto g0 =
      hpg::Gridder(hpg::Device::Serial, 0, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
    g0.set_convolution_function(hpg::Device::OpenMP, cf2);
    g0.grid_visibilities(
      hpg::Device::OpenMP,
      visibilities,
      visibility_weights,
      visibility_frequencies,
      visibility_phase,
      visibility_coordinates);
    auto norm = g0.get_normalization();
    std::cout << "normalization " << norm.real()
              << " " << norm.imag()
              << std::endl;
    auto norm0 = g0.set_normalization(norm * -1.0);
    assert(norm == norm0);
    auto norm1 = g0.get_normalization();
    std::cout << "normalization " << norm1.real()
              << " " << norm1.imag()
              << std::endl;
  }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
  std::cout << "Cuda" << std::endl;
  {
    auto st0 =
      hpg::GridderState(hpg::Device::Cuda, 2, {1000, 2000, 3}, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 =
      std::move(st2).set_convolution_function(hpg::Device::OpenMP, cf2);
  }
  {
    auto g0 = hpg::Gridder(hpg::Device::Cuda, 2, {1000, 2000, 3}, {0.1, -0.1});
    g0.fence();
    g0.fence();
    g0.set_convolution_function(hpg::Device::OpenMP, cf2);
    g0.grid_visibilities(
      hpg::Device::OpenMP,
      visibilities,
      visibility_weights,
      visibility_frequencies,
      visibility_phase,
      visibility_coordinates);
    auto norm = g0.get_normalization();
    std::cout << "normalization " << norm.real()
              << " " << norm.imag()
              << std::endl;
    auto norm0 = g0.set_normalization(norm * -1.0);
    assert(norm == norm0);
    auto norm1 = g0.get_normalization();
    std::cout << "normalization " << norm1.real()
              << " " << norm1.imag()
              << std::endl;
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
