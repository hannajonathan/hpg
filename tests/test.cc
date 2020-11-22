#include "hpg.hpp"

#include <cassert>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

const unsigned cf_oversampling = 10;

struct MyCFArray final
  : public hpg::CFArray {

  std::array<unsigned, 4> m_extent;
  std::vector<std::complex<hpg::cf_fp>> m_values;

  MyCFArray() {}

  MyCFArray(
    const std::array<unsigned, 4>& size,
    const std::vector<std::complex<hpg::cf_fp>>& values)
    : m_values(values) {

    m_extent[0] = size[0] * cf_oversampling;
    m_extent[1] = size[1] * cf_oversampling;
    m_extent[2] = size[2];
    m_extent[3] = size[3];
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
  operator()(unsigned x, unsigned y, unsigned stokes, unsigned cube)
    const override {
    return
      m_values[
        ((x * m_extent[1] + y) * m_extent[2] + stokes) * m_extent[3]
        + cube];
  }
};

template <typename Generator>
MyCFArray
create_cf(const std::array<unsigned, 4>& size, Generator& gen) {
  const unsigned num_values =
    cf_oversampling * size[0]
    * cf_oversampling * size[1]
    * size[2] * size[3];
  std::vector<std::complex<hpg::cf_fp>> values;
  values.reserve(num_values);
  std::uniform_real_distribution<hpg::cf_fp> dist(-1.0, 1.0);
  for (auto i = 0; i < num_values; ++i)
    values.emplace_back(dist(gen), dist(gen));
  return MyCFArray(size, values);
}

template <typename Generator>
void
init_visibilities(
  unsigned num_vis,
  const std::array<unsigned, 4>& grid_size,
  const std::array<float, 2>& grid_scale,
  const std::array<unsigned, 4>& cf_size,
  Generator& gen,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<unsigned>& grid_cubes,
  std::vector<unsigned>& cf_cubes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

  vis.clear();
  vis.reserve(num_vis);
  grid_cubes.clear();
  grid_cubes.reserve(num_vis);
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

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gcube(0, grid_size[3] - 1);
  std::uniform_int_distribution<unsigned> dist_gsto(0, grid_size[2] - 1);
  std::uniform_int_distribution<unsigned> dist_cfcube(0, cf_size[3] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  double ulim =
    ((cf_oversampling * (grid_size[0] - 2)) / 2
     - (cf_oversampling * cf_size[0]) / 2)
    / (grid_scale[0] * cf_oversampling * inv_lambda);
  double vlim =
    ((cf_oversampling * (grid_size[1] - 2)) / 2 -
     (cf_oversampling * cf_size[1]) / 2)
    / (grid_scale[1] * cf_oversampling * inv_lambda);
  std::uniform_real_distribution<hpg::vis_uvw_fp> dist_u(-ulim, ulim);
  std::uniform_real_distribution<hpg::vis_uvw_fp> dist_v(-vlim, vlim);
  for (auto i = 0; i < num_vis; ++i) {
    vis.emplace_back(dist_vis(gen), dist_vis(gen));
    grid_cubes.push_back(dist_gcube(gen));
    cf_cubes.push_back(dist_cfcube(gen));
    weights.push_back(dist_weight(gen));
    frequencies.push_back(freq);
    phases.emplace_back(0.0);
    coordinates.push_back(hpg::vis_uvw_t({dist_u(gen), dist_v(gen), 0.0}));
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
  const std::array<unsigned, 4>& grid_size,
  const std::array<float, 2>& grid_scale,
  const MyCFArray& cf,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<unsigned>& grid_cubes,
  std::vector<unsigned>& cf_cubes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

  {
    std::cout << "GridderState " << dev_name << std::endl;
    auto st0 =
      std::get<1>(hpg::GridderState::create(D, 2, grid_size, grid_scale));
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 =
      std::get<hpg::GridderState>(
        std::move(st2).set_convolution_function(host_dev, cf));
#ifdef HPG_ENABLE_SERIAL
    if constexpr (D == hpg::Device::Serial)
      assert(st3.max_added_tasks() == 0);
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
    if constexpr (D == hpg::Device::OpenMP)
      assert(st3.max_added_tasks() == 0);
#endif // HPG_ENABLE_OPENMP
  }
  {
    std::cout << "Gridder " << dev_name << std::endl;
    auto g0 = std::get<1>(hpg::Gridder::create(D, 2, grid_size, grid_scale));
    std::cout << "constructed" << std::endl;
    g0.set_convolution_function(host_dev, cf);
    std::cout << "cf set" << std::endl;
    g0.grid_visibilities(
      host_dev,
      vis,
      grid_cubes,
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
    auto err = g0.apply_fft();
    assert(!err);
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

template <hpg::Device D>
void
dump_grids(
  const std::string& dev_name,
  hpg::Device host_dev,
  const std::array<unsigned, 4>& grid_size,
  const std::array<float, 2>& grid_scale,
  const MyCFArray& cf,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<unsigned>& grid_cubes,
  std::vector<unsigned>& cf_cubes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates) {

  auto g0 = std::get<1>(hpg::Gridder::create(D, 2, grid_size, grid_scale));
  g0.set_convolution_function(host_dev, cf);
  g0.grid_visibilities(
    host_dev,
    vis,
    grid_cubes,
    cf_cubes,
    weights,
    frequencies,
    phases,
    coordinates);
  g0.normalize();
  auto err = g0.apply_fft();
  assert(!err);
  {
    std::cout << "after fft" << std::endl;
    auto gval = g0.grid_values();
    for (unsigned cube = 0; cube < grid_size[3]; ++cube) {
      for (unsigned sto = 0; sto < grid_size[2]; ++sto) {
        std::cout << "cube " << cube << ", sto " << sto << std::endl;
        for (unsigned y = 0; y < grid_size[1]; ++y) {
          std::cout << "  " << y << ": ";
          for (unsigned x = 0; x < grid_size[0]; ++x)
            std::cout << gval->operator()(x, y, sto, cube) << " ";
          std::cout << std::endl;
        }
      }
    }
  }
  g0.rotate_grid();
  {
    std::cout << "after rotation" << std::endl;
    auto gval = g0.grid_values();
    for (unsigned cube = 0; cube < grid_size[3]; ++cube) {
      for (unsigned sto = 0; sto < grid_size[2]; ++sto) {
        std::cout << "cube " << cube << ", sto " << sto << std::endl;
        for (unsigned y = 0; y < grid_size[1]; ++y) {
          std::cout << "  " << y << ": ";
          for (unsigned x = 0; x < grid_size[0]; ++x)
            std::cout << gval->operator()(x, y, sto, cube) << " ";
          std::cout << std::endl;
        }
      }
    }
  }
}

int
main(int argc, char* argv[]) {

  hpg::ScopeGuard hpg;

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<unsigned> cf_cubes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;

  {
    std::mt19937 rng(42);

    const std::array<unsigned, 4> grid_size{1000, 2000, 2, 1};
    const std::array<unsigned, 4> cf_size{31, 21, 2, 3};
    const std::array<float, 2> grid_scale{0.1, -0.1};
    MyCFArray cf = create_cf(cf_size, rng);
    init_visibilities(
      1000000,
      grid_size,
      grid_scale,
      cf_size,
      rng,
      vis,
      grid_cubes,
      cf_cubes,
      weights,
      frequencies,
      phases,
      coordinates);
#ifdef HPG_ENABLE_SERIAL
    run_tests<hpg::Device::Serial>(
      "Serial", hpg::Device::OpenMP,
      grid_size, grid_scale, cf,
      vis, grid_cubes, cf_cubes, weights, frequencies, phases, coordinates);
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
    run_tests<hpg::Device::Cuda>(
      "Cuda", hpg::Device::OpenMP,
      grid_size, grid_scale, cf,
      vis, grid_cubes, cf_cubes, weights, frequencies, phases, coordinates);
#endif // HPG_ENABLE_CUDA
  }
  {
    std::mt19937 rng(42);

    const std::array<unsigned, 4> grid_size{5, 6, 2, 3};
    const std::array<unsigned, 4> cf_size{3, 3, 2, 1};
    const std::array<float, 2> grid_scale{0.1, -0.1};
    MyCFArray cf = create_cf(cf_size, rng);
    init_visibilities(
      50,
      grid_size,
      grid_scale,
      cf_size,
      rng,
      vis,
      grid_cubes,
      cf_cubes,
      weights,
      frequencies,
      phases,
      coordinates);
#ifdef HPG_ENABLE_SERIAL
    dump_grids<hpg::Device::Serial>(
      "Serial", hpg::Device::OpenMP,
      grid_size, grid_scale, cf,
      vis, grid_cubes, cf_cubes, weights, frequencies, phases, coordinates);
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
    dump_grids<hpg::Device::Cuda>(
      "Cuda", hpg::Device::OpenMP,
      grid_size, grid_scale, cf,
      vis, grid_cubes, cf_cubes, weights, frequencies, phases, coordinates);
#endif // HPG_ENABLE_CUDA
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
