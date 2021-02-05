#define HPG_INTERNAL
#include "hpg.hpp"
#include "gtest/gtest.h"

#include <array>
#include <complex>
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
#endif

using namespace std::complex_literals;

struct ConeCFArray final
  : public hpg::CFArray {

  int m_oversampling;
  std::vector<unsigned> m_radius;
  std::vector<unsigned> m_oversampled_radius;

  ConeCFArray() {}

  ConeCFArray(unsigned oversampling, std::vector<unsigned> radius)
    : m_oversampling(oversampling)
    , m_radius(radius) {

    for (auto& r : radius)
      m_oversampled_radius.push_back((r /*+ padding*/) * m_oversampling);
  }

  ConeCFArray(const ConeCFArray& other)
    : m_oversampling(other.m_oversampling)
    , m_radius(other.m_radius)
    , m_oversampled_radius(other.m_oversampled_radius) {
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return m_radius.size();
  }

  std::array<unsigned, 4>
  extents(unsigned grp) const override {
    unsigned w = 2 * m_oversampled_radius[grp] + 1;
    return {w, w, 1, 1};
  }

  std::complex<float>
  operator()(unsigned x, unsigned y, unsigned, unsigned, unsigned grp)
    const override {

    std::complex<float> p(
      (-m_oversampled_radius[grp] + int(x)) + 0.5f,
      (-m_oversampled_radius[grp] + int(y)) + 0.5f);
    return
      std::polar(
        std::max(m_oversampled_radius[grp] - std::abs(p), 0.0f),
        std::arg(std::abs(p.real()) + 1.0if * std::abs(p.imag())));
  }
};

template <typename Generator>
void
init_visibilities(
  unsigned num_vis,
  const std::array<unsigned, 4>& grid_size,
  const std::array<float, 2>& grid_scale,
  const hpg::CFArray* cf,
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
  std::uniform_int_distribution<unsigned> dist_cfgrp(0, cf->num_groups() - 1);
  auto x0 = (cf->oversampling() * (grid_size[0] - 2)) / 2;
  auto y0 = (cf->oversampling() * (grid_size[1] - 2)) / 2;
  double uscale = grid_scale[0] * cf->oversampling() * inv_lambda;
  double vscale = grid_scale[1] * cf->oversampling() * inv_lambda;
  for (auto i = 0; i < num_vis; ++i) {
    auto grp = dist_cfgrp(gen);
    auto cfextents = cf->extents(grp);
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
          if ((*array)(i, j) != typename T::value_type(0))
            return true;
    } else {
    for (unsigned i = 0; i < array->extent(0); ++i)
      for (unsigned j = 0; j < array->extent(1); ++j)
        for (unsigned k = 0; k < array->extent(2); ++k)
          for (unsigned m = 0; m < array->extent(3); ++m)
            if ((*array)(i, j, k, m) != typename T::value_type(0))
              return true;
  }
  return false;
}

template <typename T>
bool
near(const T& x, const T& y) {
  return std::abs(x - y) <= 1.0e-6 * std::abs(x);
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

// test array returned from hpg::DeviceCFArray::layout_for_device()
TEST(DeviceCFArray, Create) {
  const unsigned oversampling = 20;
  ConeCFArray cf(oversampling, {10, 20, 30});
  auto arrays_or_err =
    hpg::DeviceCFArray::layout_for_device(
      default_device,
      default_host_device,
      cf);
  ASSERT_TRUE(hpg::is_value(arrays_or_err));
  auto [vsn, arrays] = hpg::get_value(std::move(arrays_or_err));
  EXPECT_EQ(arrays.size(), cf.num_groups());
  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<ConeCFArray::value_type>>>
    sized_arrays;
  for (unsigned grp = 0; grp < arrays.size(); ++grp)
    sized_arrays.emplace_back(cf.extents(grp), std::move(arrays[grp]));
  auto devcf_or_err =
    hpg::DeviceCFArray::create(vsn, oversampling, std::move(sized_arrays));
  ASSERT_TRUE(hpg::is_value(devcf_or_err));
  auto devcf = hpg::get_value(std::move(devcf_or_err));
  EXPECT_EQ(devcf->device(), default_device);
  EXPECT_EQ(devcf->num_groups(), cf.num_groups());
  EXPECT_EQ(devcf->oversampling(), cf.oversampling());
  for (unsigned grp = 0; grp < arrays.size(); ++grp)
    EXPECT_TRUE(devcf->extents(grp) == cf.extents(grp));
  for (unsigned grp = 0; grp < arrays.size(); ++grp) {
    auto radius = cf.m_oversampled_radius[grp];
    for (unsigned y = 0; y < radius; ++y)
      for (unsigned x = 0; x < radius; ++x)
        EXPECT_EQ((*devcf)(x, y, 0, 0, grp), cf(x, y, 0, 0, grp));
  }
}

// tests of layout versioning
TEST(DeviceCFArray, LayoutVersion) {
  const unsigned oversampling = 20;
  ConeCFArray cf(oversampling, {10});
  auto [vsn, arrays] =
    hpg::get_value(
      hpg::DeviceCFArray::layout_for_device(
        default_device,
        default_host_device,
        cf));
  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<ConeCFArray::value_type>>>
    sized_arrays;
  for (unsigned grp = 0; grp < arrays.size(); ++grp)
    sized_arrays.emplace_back(cf.extents(grp), std::move(arrays[grp]));
  auto devcf_or_err =
    hpg::DeviceCFArray::create(
      hpg::cf_layout_unspecified_version,
      oversampling,
      std::move(sized_arrays));
  ASSERT_TRUE(hpg::is_error(devcf_or_err));
  auto err = hpg::get_error(devcf_or_err);
  EXPECT_EQ(err.type(), hpg::ErrorType::InvalidCFLayout);
  // It would be nice to have more tests of the layout version functionality,
  // but without having access to the string format, and being limited to the
  // enabled devices makes this difficult.
}

// tests of gridding using a DeviceCFArray
TEST(DeviceCFArray, Gridding) {
  // CF definition
  const unsigned oversampling = 20;
  ConeCFArray cf(oversampling, {10});

  // create DeviceCFArray version
  auto [vsn, arrays] =
    hpg::get_value(
      hpg::DeviceCFArray::layout_for_device(
        default_device,
        default_host_device,
        cf));
  std::vector<
    std::tuple<std::array<unsigned, 4>, std::vector<ConeCFArray::value_type>>>
    sized_arrays;
  for (unsigned grp = 0; grp < arrays.size(); ++grp)
    sized_arrays.emplace_back(cf.extents(grp), std::move(arrays[grp]));
  auto devcf =
    hpg::get_value(
      hpg::DeviceCFArray::create(vsn, oversampling, std::move(sized_arrays)));

  // grid definition
  std::array<unsigned, 4> grid_size{50, 50, 1, 1};
  std::array<float, 2> grid_scale{0.1, -0.1};

  // visibilities
  unsigned num_vis = 1000;
  std::mt19937 rng(42);
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
    &cf,
    rng,
    vis,
    grid_cubes,
    cf_indexes,
    weights,
    frequencies,
    phases,
    coordinates,
    cf_phase_screens);

  // cf GridderState
  auto gs_cf =
    hpg::get_value(
      hpg::flatmap(
        hpg::GridderState::create(
          default_device,
          0,
          num_vis,
          &cf,
          grid_size,
          grid_scale),
        [&cf](auto&& gs) {
          return
            std::move(gs)
            .set_convolution_function(default_host_device, std::move(cf));
        }));

  // devcf GridderState
  auto gs_devcf =
    hpg::get_value(
      hpg::flatmap(
        hpg::GridderState::create(
          default_device,
          0,
          num_vis,
          devcf.get(),
          grid_size,
          grid_scale),
        [&devcf](auto&& gs) {
          return
            std::move(gs)
            .set_convolution_function(
              default_host_device,
              std::move(*devcf));
        }));

  // function to grid visibilities and return gridded values
  auto gridding =
    hpg::RvalM<const hpg::GridderState&, hpg::GridderState>::pure(
      [&](const hpg::GridderState& gs) {
        return
          gs
          .grid_visibilities(
            default_host_device,
            decltype(vis)(vis),
            decltype(grid_cubes)(grid_cubes),
            decltype(cf_indexes)(cf_indexes),
            decltype(weights)(weights),
            decltype(frequencies)(frequencies),
            decltype(phases)(phases),
            decltype(coordinates)(coordinates),
            decltype(cf_phase_screens)(cf_phase_screens));
      })
    .map(
      [](auto&& gs) {
        return std::get<1>(std::move(gs).grid_values());
      });

  auto grid_cf = hpg::get_value(gridding(gs_cf));
  ASSERT_TRUE(has_non_zero(grid_cf.get()));
  auto grid_devcf_or_err = gridding(gs_devcf);
  ASSERT_TRUE(hpg::is_value(grid_devcf_or_err));
  auto grid_devcf = hpg::get_value(std::move(grid_devcf_or_err));
  EXPECT_TRUE(values_eq(grid_cf.get(), grid_devcf.get()));
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
