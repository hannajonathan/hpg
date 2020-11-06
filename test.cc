#include "hpg.hpp"

#include <cassert>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

const std::array<unsigned, 4> grid_size{1000, 2000, 2, 1};
const std::array<unsigned, 4> cf_size{31, 21, 2, 3};
const unsigned cf_oversampling = 10;

struct MyCFArray final
  : public hpg::CFArray {

  std::array<unsigned, 4> m_extent;
  std::vector<std::complex<hpg::cf_fp>> m_values;

  MyCFArray() {}

  MyCFArray(const std::vector<std::complex<hpg::cf_fp>>& values)
    : m_values(values) {

    m_extent[0] = cf_size[0] * cf_oversampling;
    m_extent[1] = cf_size[1] * cf_oversampling;
    m_extent[2] = cf_size[2];
    m_extent[3] = cf_size[3];
  }

  unsigned
  oversampling() const override {
    return cf_oversampling;
  }

  unsigned
  extent(unsigned dim) const override {
    return m_extent[dim];
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned polarization, unsigned cube)
    const override {
    return
      m_values[
        ((x * m_extent[1] + y) * m_extent[2] + polarization) * m_extent[3]
        + cube];
  }
};

template <typename Generator>
MyCFArray
create_cf(Generator& gen) {
  const unsigned num_values =
    cf_oversampling * cf_size[0]
    * cf_oversampling * cf_size[1]
    * cf_size[2] * cf_size[3];
  std::vector<std::complex<hpg::cf_fp>> values;
  values.reserve(num_values);
  std::uniform_real_distribution<hpg::cf_fp> dist(-1.0, 1.0);
  for (auto i = 0; i < num_values; ++i)
    values.emplace_back(dist(gen), dist(gen));
  return MyCFArray(values);
}

template <typename Generator>
void
init_visibilities(
  Generator& gen,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<hpg::grid_plane_t>& grid_planes,
  std::vector<unsigned>& cf_cubes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

  const unsigned num_vis = 1000000;
  vis.clear();
  vis.reserve(num_vis);
  grid_planes.clear();
  grid_planes.reserve(num_vis);
  cf_cubes.clear();
  cf_cubes.reserve(num_vis);
  weights.clear();
  weights.reserve(num_vis);
  frequencies.clear();
  frequencies.reserve(num_vis);
  phases.clear();
  phases.reserve(num_vis);
  coordinates.clear();
  coordinates.reserve(num_vis);

  assert(grid_size[3] == 1);
  const unsigned gcube = 0;
  std::uniform_int_distribution<unsigned> dist_gsto(0, grid_size[2] - 1);
  std::uniform_int_distribution<unsigned> dist_cfcube(0, cf_size[3] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  std::uniform_real_distribution<hpg::vis_uvw_fp> dist_uvw(-475.0, 475.0);
  for (auto i = 0; i < num_vis; ++i) {
    vis.emplace_back(dist_vis(gen), dist_vis(gen));
    grid_planes.push_back({dist_gsto(gen), gcube});
    cf_cubes.push_back(dist_cfcube(gen));
    weights.push_back(dist_weight(gen));
    frequencies.push_back(2.92513197327302e9);
    phases.emplace_back(0.0);
    coordinates.push_back(hpg::vis_uvw_t({dist_uvw(gen), dist_uvw(gen), 0.0}));
  }
}

std::complex<hpg::grid_value_fp>
sum_grid(const hpg::GridValueArray* array) {
  std::complex<hpg::grid_value_fp> result;
  for (auto i = 0; i < array->extent(0); ++i)
    for (auto j = 0; j < array->extent(1); ++j)
      for (auto k = 0; k < array->extent(2); ++k)
        for (auto m = 0; m < array->extent(3); ++m)
          result += array->operator()(i, j, k, m);
  return result;
}

template <hpg::Device D>
void
run_tests(
  const std::string& dev_name,
  hpg::Device host_dev,
  const MyCFArray& cf,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<hpg::grid_plane_t>& grid_planes,
  std::vector<unsigned>& cf_cubes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

  {
    std::cout << "GridderState " << dev_name << std::endl;
    auto st0 = hpg::GridderState(D, 4, grid_size, {0.1, -0.1});
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 = std::move(st2).set_convolution_function(host_dev, cf);
    assert(st3.max_async_tasks() == 0);
  }
  {
    std::cout << "Gridder " << dev_name << std::endl;
    auto g0 = hpg::Gridder(D, 0, grid_size, {0.1, -0.1});
    std::cout << "constructed" << std::endl;
    g0.set_convolution_function(host_dev, cf);
    std::cout << "cf set" << std::endl;
    g0.grid_visibilities(
      host_dev,
      vis,
      grid_planes,
      cf_cubes,
      weights,
      frequencies,
      phases,
      coordinates);
    std::cout << "gridded" << std::endl;
    auto weights = g0.grid_weights();
    std::cout << "weights";
    for (auto sto = 0; sto < grid_size[2]; ++sto)
      for (auto ch = 0; ch < grid_size[3]; ++ch)
        std::cout << " " << weights->operator()(sto, ch);
    std::cout << std::endl;
    {
      auto grid = g0.grid_values();
      auto sum = sum_grid(grid.get());
      std::cout << "sum " << sum << std::endl;
    }
    g0.normalize();
    std::cout << "grid normalized" << std::endl;
    {
      auto grid = g0.grid_values();
      auto sum = sum_grid(grid.get());
      std::cout << "sum " << sum << std::endl;
    }
    g0.apply_fft();
    std::cout << "fft applied" << std::endl;
    g0.reset_grid();
    std::cout << "grid reset" << std::endl;
    {
      auto grid = g0.grid_values();
      auto sum = sum_grid(grid.get());
      std::cout << "sum " << sum << std::endl;
    }
  }
}

int
main(int argc, char* argv[]) {

  hpg::ScopeGuard hpg;

  std::mt19937 rng(42);

  MyCFArray cf = create_cf(rng);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<hpg::grid_plane_t> grid_planes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;
  init_visibilities(
    rng,
    vis,
    grid_planes,
    cf_cubes,
    weights,
    frequencies,
    phases,
    coordinates);

#ifdef HPG_ENABLE_SERIAL
  run_tests<hpg::Device::Serial>(
    "Serial", hpg::Device::OpenMP, cf,
    vis, grid_planes, cf_cubes, weights, frequencies, phases, coordinates);
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
  run_tests<hpg::Device::Cuda>(
    "Cuda", hpg::Device::OpenMP, cf,
    vis, grid_planes, cf_cubes, weights, frequencies, phases, coordinates);
#endif // HPG_ENABLE_CUDA
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
