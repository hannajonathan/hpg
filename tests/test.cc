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
  num_groups() const override {
    return 1;
  }

  std::array<unsigned, 4>
  extents(unsigned) const override {
    return m_extent;
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned mueller, unsigned cube, unsigned)
    const override {
    return
      m_values[
        ((x * m_extent[1] + y) * m_extent[2] + mueller) * m_extent[3] + cube];
  }
};

template <typename Generator>
MyCFArray
create_cf(const std::array<unsigned, 4>& size, Generator& gen) {
  const unsigned num_values =
    cf_oversampling * size[0]
    * cf_oversampling * size[1]
    * size[2] * size[4];
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
  std::vector<hpg::VisData<1>>& vis) {

  vis.clear();
  vis.reserve(num_vis);

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gcube(0, grid_size[3] - 1);
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
    vis.push_back(
      hpg::VisData<1>(
        {std::complex<hpg::visibility_fp>(dist_vis(gen), dist_vis(gen))},
        {dist_weight(gen)},
        freq,
        0.0,
        hpg::vis_uvw_t({dist_u(gen), dist_v(gen), 0.0}),
        dist_gcube(gen),
        {dist_cfcube(gen), 0}));
  }
}

std::complex<hpg::grid_value_fp>
sum_grid(const hpg::GridValueArray* array) {
  std::complex<hpg::grid_value_fp> result;
  for (auto i = 0; i < array->extent(0); ++i)
    for (auto j = 0; j < array->extent(1); ++j)
      for (auto k = 0; k < array->extent(2); ++k)
        for (auto m = 0; m < array->extent(3); ++m)
          result += (*array)(i, j, k, m);
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
  std::vector<hpg::VisData<1>>& vis) {

  std::vector<std::array<int, 1>> mueller_indexes;
  for (size_t i = 0; i < grid_size[2]; ++i)
    mueller_indexes.push_back({0});

  {
    std::cout << "GridderState " << dev_name << std::endl;
    auto st0 =
      std::get<1>(
        hpg::GridderState::create<1>(
          D,
          2,
          vis.size(),
          &cf,
          grid_size,
          grid_scale,
          mueller_indexes,
          mueller_indexes));
    auto st1 = st0.fence();
    auto st2 = std::move(st0).fence();
    auto st3 =
      std::get<hpg::GridderState>(
        std::move(st2).set_convolution_function(host_dev, MyCFArray(cf)));
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
    auto g0 =
      std::get<1>(
        hpg::Gridder
        ::create<1>(
          D,
          2,
          vis.size(),
          &cf,
          grid_size,
          grid_scale,
          mueller_indexes,
          mueller_indexes));
    std::cout << "constructed" << std::endl;
    g0.set_convolution_function(host_dev, MyCFArray(cf));
    std::cout << "cf set" << std::endl;
    g0.grid_visibilities(host_dev, std::remove_reference_t<decltype(vis)>(vis));
    std::cout << "gridded" << std::endl;
    auto weights = g0.grid_weights();
    std::cout << "weights";
    for (auto copol = 0; copol < grid_size[2]; ++copol)
      for (auto ch = 0; ch < grid_size[3]; ++ch)
        std::cout << " " << weights->operator()(copol, ch);
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
    auto err = g0.apply_grid_fft();
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
  std::vector<hpg::VisData<1>>& vis) {

  std::vector<std::array<int, 1>> mueller_indexes;
  for (size_t i = 0; i < grid_size[2]; ++i)
    mueller_indexes.push_back({0});

  auto g0 =
    std::get<1>(
      hpg::Gridder
      ::create<1>(
        D,
        2,
        vis.size(),
        &cf,
        grid_size,
        grid_scale,
        mueller_indexes,
        mueller_indexes));
  g0.set_convolution_function(host_dev, MyCFArray(cf));
  g0.grid_visibilities(
    host_dev,
    std::remove_reference_t<decltype(vis)>(vis));
  g0.normalize();
  auto err = g0.apply_grid_fft();
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
            std::cout << (*gval)(x, y, sto, cube) << " ";
          std::cout << std::endl;
        }
      }
    }
  }
  g0.shift_grid();
  {
    std::cout << "after rotation" << std::endl;
    auto gval = g0.grid_values();
    for (unsigned cube = 0; cube < grid_size[3]; ++cube) {
      for (unsigned sto = 0; sto < grid_size[2]; ++sto) {
        std::cout << "cube " << cube << ", sto " << sto << std::endl;
        for (unsigned y = 0; y < grid_size[1]; ++y) {
          std::cout << "  " << y << ": ";
          for (unsigned x = 0; x < grid_size[0]; ++x)
            std::cout << (*gval)(x, y, sto, cube) << " ";
          std::cout << std::endl;
        }
      }
    }
  }
}

int
main(int argc, char* argv[]) {

  hpg::ScopeGuard hpg;

  std::vector<hpg::VisData<1>> vis;

  {
    std::mt19937 rng(42);

    const std::array<unsigned, 4> grid_size{1000, 2000, 2, 1};
    const std::array<unsigned, 4> cf_size{31, 21, 1, 3};
    const std::array<float, 2> grid_scale{0.1, -0.1};
    MyCFArray cf = create_cf(cf_size, rng);
    const unsigned num_visibilities = 1000000;
    init_visibilities(
      num_visibilities,
      grid_size,
      grid_scale,
      cf_size,
      rng,
      vis);
#ifdef HPG_ENABLE_SERIAL
    run_tests<hpg::Device::Serial>(
      "Serial", hpg::Device::OpenMP,
      grid_size, grid_scale, cf, vis);
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
    run_tests<hpg::Device::Cuda>(
      "Cuda", hpg::Device::OpenMP,
      grid_size, grid_scale, cf, vis);
#endif // HPG_ENABLE_CUDA
  }
  {
    std::mt19937 rng(42);

    const std::array<unsigned, 4> grid_size{5, 6, 2, 3};
    const std::array<unsigned, 4> cf_size{3, 3, 1, 2};
    const std::array<float, 2> grid_scale{0.1, -0.1};
    MyCFArray cf = create_cf(cf_size, rng);
    const unsigned num_visibilities = 50;
    init_visibilities(
      num_visibilities,
      grid_size,
      grid_scale,
      cf_size,
      rng,
      vis);
#ifdef HPG_ENABLE_SERIAL
    dump_grids<hpg::Device::Serial>(
      "Serial", hpg::Device::OpenMP,
      grid_size, grid_scale, cf, vis);
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_CUDA
    dump_grids<hpg::Device::Cuda>(
      "Cuda", hpg::Device::OpenMP,
      grid_size, grid_scale, cf, vis);
#endif // HPG_ENABLE_CUDA
  }
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
