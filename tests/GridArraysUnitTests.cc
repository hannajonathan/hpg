#define HPG_INTERNAL
#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <cassert>
#include <complex>
#include <experimental/array>
#include <iostream>
#include <random>

#if defined(HPG_ENABLE_OPENMP)
hpg::Device default_host_device = hpg::Device::OpenMP;
#elif defined(HPG_ENABLE_SERIAL)
hpg::Device default_host_device = hpg::Device::Serial;
#else
# error "At least one host device needs to be enabled"
#endif

#if defined(HPG_ENABLE_CUDA)
hpg::Device default_device = hpg::Device::Cuda;
#elif defined(HPG_ENABLE_OPENMP)
hpg::Device default_device = hpg::Device::OpenMP;
#elif defined(HPG_ENABLE_SERIAL)
hpg::Device default_device = hpg::Device::Serial;
#else
# error "At least one device needs to be enabled"
#endif

using namespace std::complex_literals;

template <typename T>
struct epsilon {
  using F = void;
  // static constexpr F val:
};
template <>
struct epsilon<float> {
  using F = float;
  static constexpr float val = 1.0e-3;
};
template <>
struct epsilon<double> {
  using F = double;
  static constexpr double val = 1.0e-3;
};
template <>
struct epsilon<std::complex<float>> {
  using F = float;
  static constexpr float val = 1.0e-3;
};
template <>
struct epsilon<std::complex<double>> {
  using F = double;
  static constexpr double val = 1.0e-3;
};

template <typename T>
bool
near(const T& x, const T& y) {
  return std::abs(x - y) <= epsilon<T>::val * std::abs(x);
}
template <>
bool
near(const std::complex<double>& x, const std::complex<double>& y) {
  auto d = x - y;
  return
    std::abs(d.real()) <= epsilon<double>::val * std::abs(x.real())
    && std::abs(d.imag()) <= epsilon<double>::val * std::abs(x.imag());
}
template <>
bool
near(const std::complex<float>& x, const std::complex<float>& y) {
  auto d = x - y;
  return
    std::abs(d.real()) <= epsilon<float>::val * std::abs(x.real())
    && std::abs(d.imag()) <= epsilon<float>::val * std::abs(x.imag());
}

struct MyCFArrayShape
  : public hpg::CFArrayShape {

  unsigned m_oversampling;
  std::vector<std::array<unsigned, 4>> m_extents;

  MyCFArrayShape() {}

  MyCFArrayShape(
    unsigned oversampling,
    const std::vector<std::array<unsigned, 4>>& sizes)
    : m_oversampling(oversampling) {

    for (auto& sz : sizes)
      m_extents.push_back(
        {sz[0] * oversampling, sz[1] * oversampling, sz[2], sz[3]});
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return static_cast<unsigned>(m_extents.size());
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }
};

struct MyCFArray final
  : public hpg::CFArray {

  unsigned m_oversampling;
  std::vector<std::array<unsigned, 4>> m_extents;
  std::vector<std::vector<std::complex<hpg::cf_fp>>> m_values;

  MyCFArray() {}

  MyCFArray(
    unsigned oversampling,
    const std::vector<std::array<unsigned, 4>>& sizes,
    const std::vector<std::vector<std::complex<hpg::cf_fp>>>& values)
    : m_oversampling(oversampling)
    , m_values(values) {

    for (auto& sz : sizes)
      m_extents.push_back(
        {sz[0] * oversampling, sz[1] * oversampling, sz[2], sz[3]});
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return static_cast<unsigned>(m_extents.size());
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }

  std::complex<hpg::cf_fp>
  operator()(unsigned x, unsigned y, unsigned copol, unsigned cube, unsigned grp)
    const override {
    auto& vals = m_values[grp];
    auto& ext = m_extents[grp];
    return vals[((x * ext[1] + y) * ext[2] + copol) * ext[3] + cube];
  }
};

template <typename Generator>
MyCFArray
create_cf(
  unsigned oversampling,
  const std::vector<std::array<unsigned, 4>>& sizes,
  Generator& gen) {

  std::vector<std::vector<std::complex<hpg::cf_fp>>> values;
  for (auto& sz : sizes) {
    decltype(values)::value_type vs;
    const unsigned num_values =
      oversampling * sz[0] * oversampling * sz[1] * sz[2] * sz[3];
    vs.reserve(num_values);
    std::uniform_real_distribution<hpg::cf_fp> dist(-1.0, 1.0);
    for (auto i = 0; i < num_values; ++i)
      vs.emplace_back(dist(gen), dist(gen));
    values.push_back(std::move(vs));
  }
  return MyCFArray(oversampling, sizes, values);
}

template <typename Generator>
void
init_visibilities(
  unsigned num_vis,
  const std::array<unsigned, 4>& grid_size,
  const std::array<hpg::grid_scale_fp, 2>& grid_scale,
  const MyCFArray& cf,
  Generator& gen,
  std::vector<std::complex<hpg::visibility_fp>>& vis,
  std::vector<unsigned>& grid_cubes,
  std::vector<hpg::vis_cf_index_t>& cf_indexes,
  std::vector<hpg::vis_weight_fp>& weights,
  std::vector<hpg::vis_frequency_fp>& frequencies,
  std::vector<hpg::vis_phase_fp>& phases,
  std::vector<hpg::vis_uvw_t>& coordinates,
  std::vector<hpg::cf_phase_screen_t>& cf_phase_screens) {

  vis.clear();
  vis.reserve(num_vis);
  grid_cubes.clear();
  grid_cubes.reserve(num_vis);
  cf_indexes.clear();
  cf_indexes.reserve(num_vis);
  weights.clear();
  weights.reserve(num_vis);
  frequencies.clear();
  frequencies.reserve(num_vis);
  phases.clear();
  phases.reserve(num_vis);
  coordinates.clear();
  coordinates.reserve(num_vis);
  cf_phase_screens.reserve(num_vis);

  const double inv_lambda = 9.75719;
  const double freq = 299792458.0 * inv_lambda;
  std::uniform_int_distribution<unsigned> dist_gcube(0, grid_size[3] - 1);
  std::uniform_int_distribution<unsigned> dist_gcopol(0, grid_size[2] - 1);
  std::uniform_real_distribution<hpg::visibility_fp> dist_vis(-1.0, 1.0);
  std::uniform_real_distribution<hpg::vis_weight_fp> dist_weight(0.0, 1.0);
  std::uniform_real_distribution<hpg::cf_phase_screen_fp>
    dist_cfscreen(-3.141, 3.141);
  std::uniform_int_distribution<unsigned> dist_cfgrp(0, cf.num_groups() - 1);
  auto x0 = (cf.oversampling() * (grid_size[0] - 2)) / 2;
  auto y0 = (cf.oversampling() * (grid_size[1] - 2)) / 2;
  double uscale = grid_scale[0] * cf.oversampling() * inv_lambda;
  double vscale = grid_scale[1] * cf.oversampling() * inv_lambda;
  for (auto i = 0; i < num_vis; ++i) {
    auto grp = dist_cfgrp(gen);
    auto cfextents = cf.extents(grp);
    std::uniform_int_distribution<unsigned> dist_cfcube(0, cfextents[3] - 1);
    double ulim = (x0 - (cfextents[0]) / 2) / uscale;
    double vlim = (y0 - (cfextents[1]) / 2) / vscale;
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_u(-ulim, ulim);
    std::uniform_real_distribution<hpg::vis_uvw_fp> dist_v(-vlim, vlim);
    vis.emplace_back(dist_vis(gen), dist_vis(gen));
    grid_cubes.push_back(dist_gcube(gen));
    cf_indexes.push_back({dist_cfcube(gen), grp});
    weights.push_back(dist_weight(gen));
    frequencies.push_back(freq);
    phases.emplace_back(0.0);
    coordinates.push_back(hpg::vis_uvw_t({dist_u(gen), dist_v(gen), 0.0}));
    cf_phase_screens.push_back({dist_cfscreen(gen), dist_cfscreen(gen)});
  }
}

template <typename T>
bool
has_non_zero(const T* array) {
  if constexpr (T::rank == 2) {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        if ((*array)(i, j) != typename T::scalar_type(0))
          return true;
  } else {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        for (unsigned k = 0; k < array->extent(2); ++k)
          for (unsigned m = 0; m < array->extent(3); ++m)
            if ((*array)(i, j, k, m) != typename T::scalar_type(0))
              return true;
  }
  return false;
}

template <typename T>
bool
values_eq(const T* array0, const T* array1) {
  if constexpr (T::rank == 2) {
    if (array0->extent(0) != array1->extent(0)
        || array0->extent(1) != array1->extent(1)) {
      std::cerr << "extents differ" << std::endl;
      return false;
    }
    for (unsigned i = 0; i < array0->extent(0); ++i)
      for (unsigned j = 0; j < array0->extent(1); ++j)
        if (!near((*array0)(i, j), (*array1)(i, j))) {
          std::cerr << "values differ at "
                    << i << "," << j
                    << "; " << (*array0)(i, j)
                    << " != " << (*array1)(i, j)
                    << std::endl;
          return false;
        }
    } else {
    if (array0->extent(0) != array1->extent(0)
        || array0->extent(1) != array1->extent(1)
        || array0->extent(2) != array1->extent(2)
        || array0->extent(3) != array1->extent(3)) {
      std::cerr << "extents differ" << std::endl;
      return false;
    }
    for (unsigned i = 0; i < array0->extent(0); ++i)
      for (unsigned j = 0; j < array0->extent(1); ++j)
        for (unsigned k = 0; k < array0->extent(2); ++k)
          for (unsigned m = 0; m < array0->extent(3); ++m)
            if (!near((*array0)(i, j, k, m), (*array1)(i, j, k, m))) {
              std::cerr << "values differ at "
                        << i << "," << j << "," << k << "," << m
                        << "; " << (*array0)(i, j, k, m)
                        << " != " << (*array1)(i, j, k, m)
                        << std::endl;
              return false;
            }
  }
  return true;
}

TEST(GridValueArray, CopyIntoLayouts) {
  std::array<unsigned, 4> grid_size{20, 20, 4, 3};
  std::array<hpg::grid_scale_fp, 2> grid_scale{0.1, -0.1};
  size_t num_vis = 100;
  auto padding = 2 * hpg::CFArray::padding;
  const std::vector<std::array<unsigned, 4>>
    cf_sizes{
      {3 + padding, 3 + padding, 4, 3},
      {2 + padding, 2 + padding, 4, 2}};

  std::mt19937 rng(42);
  MyCFArray cf = create_cf(10, cf_sizes, rng);

  std::vector<std::complex<hpg::visibility_fp>> vis;
  std::vector<unsigned> grid_cubes;
  std::vector<hpg::vis_cf_index_t> cf_indexes;
  std::vector<hpg::vis_weight_fp> weights;
  std::vector<hpg::vis_frequency_fp> frequencies;
  std::vector<hpg::vis_phase_fp> phases;
  std::vector<hpg::vis_uvw_t> coordinates;
  std::vector<hpg::cf_phase_screen_t> cf_phase_screens;
  init_visibilities(
    num_vis,
    grid_size,
    grid_scale,
    cf,
    rng,
    vis,
    grid_cubes,
    cf_indexes,
    weights,
    frequencies,
    phases,
    coordinates,
    cf_phase_screens);

  auto gridding =
    hpg::RvalM<void, hpg::GridderState>::pure(
      [&]() {
        return
          hpg::GridderState::create(
            default_device,
            0,
            num_vis,
            &cf,
            grid_size,
            grid_scale);
      })
    .and_then(
      [&](auto&& gs) mutable {
        return
          std::move(gs)
          .set_convolution_function(default_host_device, std::move(cf));
      })
    .and_then(
      [&](auto&& gs) mutable {
        return
          std::move(gs)
          .grid_visibilities(
            default_host_device,
            std::move(vis),
            std::move(grid_cubes),
            std::move(cf_indexes),
            std::move(weights),
            std::move(frequencies),
            std::move(phases),
            std::move(coordinates),
            std::move(cf_phase_screens));
      })
    .map(
      [](auto&& gs) {
        return std::get<1>(std::move(gs).grid_values());
      });

  auto grid_or_err = gridding();
  ASSERT_TRUE(hpg::is_value(grid_or_err));
  auto grid = hpg::get_value(std::move(grid_or_err));
  ASSERT_TRUE(has_non_zero(grid.get()));
  auto sz = grid->min_buffer_size();
  std::vector<hpg::GridValueArray::scalar_type> grid_left(sz);
  std::vector<hpg::GridValueArray::scalar_type> grid_right(sz);
  grid->copy_into(default_host_device, grid_left.data(), hpg::Layout::Left);
  grid->copy_into(default_host_device, grid_right.data(), hpg::Layout::Right);
  bool eq = true;
  for (unsigned x = 0; eq && x < grid_size[0]; ++x)
    for (unsigned y = 0; eq && y < grid_size[1]; ++y)
      for (unsigned mr = 0; eq && mr < grid_size[2]; ++mr)
        for (unsigned cb = 0; eq && cb < grid_size[3]; ++cb) {
          auto& l =
            grid_left[
              x + grid_size[0] * (y + grid_size[1] * (mr + grid_size[2] * cb))];
          auto& r =
            grid_right[
              cb + grid_size[3] * (mr + grid_size[2] * (y + grid_size[1] * x))];
          EXPECT_EQ(l, r);
          eq = (l == r) && (l == (*grid)(x, y, mr, cb));
          if (!eq)
            std::cout << "at "
                      << x << ","
                      << y << ","
                      << mr << ","
                      << cb << ": "
                      << l << " "
                      << r << " "
                      << (*grid)(x, y, mr, cb) << std::endl;
        }
}

int
main(int argc, char **argv) {
  std::ostringstream oss;
  oss << "Using ";
  switch (default_device) {
#ifdef HPG_ENABLE_SERIAL
  case hpg::Device::Serial:
    oss << "Serial";
    break;
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
  case hpg::Device::OpenMP:
    oss << "OpenMP";
    break;
#endif // HPG_ENABLE_OPENMP
#ifdef HPG_ENABLE_CUDA
  case hpg::Device::Cuda:
    oss << "Cuda";
    break;
#endif // HPG_ENABLE_CUDA
  default:
    assert(false);
    break;
  }
  oss << " device for tests";
  std::cout << oss.str() << std::endl;

  ::testing::InitGoogleTest(&argc, argv);
  int rc;
  {
    // weird, but using ScopeGuard/initialize/finalize at function scope can
    // sometimes hang this program on exit (but not when executed by ctest)
    hpg::initialize();
    rc = RUN_ALL_TESTS();
    hpg::finalize();
  }
  return rc;
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
