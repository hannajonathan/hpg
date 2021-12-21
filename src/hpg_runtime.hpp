// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once

#include "hpg_config.hpp"
#include "hpg_impl.hpp"
// #include "hpg_export.h"

#include <any>
#include <deque>
#include <mutex>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace hpg::runtime {

/** helper type for std::visit */
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
/** explicit deduction guide (not needed as of C++20) */
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace K = Kokkos;

/** abstract base class for state implementations */
struct /*HPG_EXPORT*/ State {

  using maybe_vis_t =
    std::shared_ptr<std::shared_ptr<std::optional<VisDataVector>>>;

  virtual Device
  device() const noexcept = 0;

  virtual unsigned
  visibility_gridder_version() const noexcept = 0;

  virtual unsigned
  grid_normalizer_version() const noexcept = 0;

  virtual unsigned
  fft_version() const noexcept = 0;

  virtual unsigned
  grid_shifter_version() const noexcept = 0;

  virtual unsigned
  max_active_tasks() const noexcept = 0;

  virtual size_t
  visibility_batch_size() const noexcept = 0;

  virtual size_t
  max_avg_channels_per_vis() const noexcept = 0;

  virtual std::array<unsigned, 4>
  grid_size_global() const noexcept = 0;

  virtual std::array<unsigned, 4>
  grid_size_local() const noexcept = 0;

  virtual std::array<unsigned, 4>
  grid_offset_local() const noexcept = 0;

  virtual std::array<grid_scale_fp, 2>
  grid_scale() const noexcept = 0;

  virtual unsigned
  num_polarizations() const noexcept = 0;

  virtual size_t
  convolution_function_region_size(const CFArrayShape* shape)
    const noexcept = 0;

  virtual std::optional<std::unique_ptr<Error>>
  allocate_convolution_function_region(
    unsigned context,
    const CFArrayShape* shape) = 0;

  virtual std::optional<std::unique_ptr<Error>>
  set_convolution_function(
    unsigned context,
    Device host_device,
    CFArray&& cf) = 0;

  virtual std::optional<std::unique_ptr<Error>>
  set_model(Device host_device, GridValueArray&& gv) = 0;

  virtual std::variant<std::unique_ptr<Error>, maybe_vis_t>
  grid_visibilities(
    unsigned context,
    Device host_device,
    VisDataVector&& visibilities,
    std::vector<vis_weight_fp>&& wgt_values,
    std::vector<unsigned>&& wgt_col_index,
    std::vector<size_t>&& wgt_row_index,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) = 0;

  virtual void
  fence() const noexcept = 0;

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const = 0;

  virtual std::shared_ptr<GridWeightArray::value_type>
  grid_weights_ptr() const = 0;

  virtual size_t
  grid_weights_span() const = 0;

  virtual std::unique_ptr<GridValueArray>
  grid_values() const = 0;

  virtual std::shared_ptr<GridValueArray::value_type>
  grid_values_ptr() const = 0;

  virtual size_t
  grid_values_span() const = 0;

  virtual void
  reset_grid() = 0;

  virtual void
  fill_grid(const impl::gv_t& val) = 0;

  virtual void
  fill_grid_weights(const grid_value_fp& val) = 0;

  virtual std::unique_ptr<GridValueArray>
  model_values() const = 0;

  virtual std::shared_ptr<GridValueArray::value_type>
  model_values_ptr() const = 0;

  virtual size_t
  model_values_span() const = 0;

  virtual void
  reset_model() = 0;

  virtual void
  normalize_by_weights(grid_value_fp wfactor) = 0;

  virtual std::optional<std::unique_ptr<Error>>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place) = 0;

  virtual std::optional<std::unique_ptr<Error>>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place) = 0;

  virtual void
  shift_grid(ShiftDirection direction) = 0;

  virtual void
  shift_model(ShiftDirection direction) = 0;

  virtual ~State() {}
};

/** type trait associating Kokkos device with hpg Device */
template <typename Device>
struct /*HPG_EXPORT*/ DeviceTraits {
  static constexpr unsigned active_task_limit = 0;

  using stream_type = void;
};

#ifdef HPG_ENABLE_SERIAL
/** Serial device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceTraits<K::Serial> {
  static constexpr unsigned active_task_limit = 1;

  using stream_type = void;
};
#endif // HPG_ENABLE_SERIAL

#ifdef HPG_ENABLE_OPENMP
/** OpenMP device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceTraits<K::OpenMP> {
  static constexpr unsigned active_task_limit = 1;

  using stream_type = void;
};
#endif // HPG_ENABLE_OPENMP

#ifdef HPG_ENABLE_CUDA
/** Cuda device type trait */
template <>
struct /*HPG_EXPORT*/ DeviceTraits<K::Cuda> {
  // the maximum number of concurrent kernels for NVIDIA devices depends on
  // compute capability; set a large value here, much larger than any capability
  // through 8.6, and leave it to the user to limit the request
  static constexpr unsigned active_task_limit = 1024;

  using stream_type = cudaStream_t;

  static bool
  create_stream(stream_type& stream) {
    auto rc = cudaStreamCreate(&stream);
    return rc == cudaSuccess;
  }

  static bool
  destroy_stream(stream_type& stream) {
    bool result = true;
    if (stream) {
      auto rc = cudaStreamDestroy(stream);
      result = rc == cudaSuccess;
      stream = NULL;
    }
    return result;
  }
};
#endif // HPG_ENABLE_CUDA

/** names for stream states */
enum class /*HPG_EXPORT*/ StreamPhase {
  PRE_DEGRIDDING,
  DEGRIDDING,
  PRE_GRIDDING,
  GRIDDING,
};

/** formatted output for StreamPhase value */
std::ostream&
operator<<(std::ostream& str, const StreamPhase& ph);

template <Device D>
struct StateT;

/** memory pool and views therein for elements of a (sparse) CF */
template <Device D>
struct /*HPG_EXPORT*/ CFPool final {

  using kokkos_device = typename impl::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using cfd_view =
    impl::cf_view<
      typename impl::CFLayout<kokkos_device>::layout,
      memory_space>;
  using cfh_view = typename cfd_view::HostMirror;

  StateT<D> *state;
  K::View<impl::cf_t*, memory_space> pool;
  unsigned num_cf_groups;
  unsigned max_cf_extent_y;
  K::Array<cfd_view, HPG_MAX_NUM_CF_GROUPS> cf_d; // unmanaged (in pool)
  std::vector<std::any> cf_h;
  K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS> cf_radii;

  CFPool()
    : state(nullptr)
    , num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(StateT<D>* st)
    : state(st)
    , num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(const CFPool& other) = delete;

  CFPool(CFPool&& other) noexcept {

    if (other.state)
      other.state->fence_unlocked();
    std::swap(pool, other.pool);
    std::swap(num_cf_groups, other.num_cf_groups);
    std::swap(max_cf_extent_y, other.max_cf_extent_y);
    std::swap(cf_d, other.cf_d);
    std::swap(cf_h, other.cf_h);
    std::swap(cf_radii, other.cf_radii);
  }

  CFPool&
  operator=(const CFPool& rhs) {

    // TODO: make this exception-safe?
    reset();
    num_cf_groups = rhs.num_cf_groups;
    max_cf_extent_y = rhs.max_cf_extent_y;
    cf_radii = rhs.cf_radii;
    if (rhs.pool.extent(0) > 0) {
      pool =
        decltype(pool)(
          K::ViewAllocateWithoutInitializing("cf"),
          rhs.pool.extent(0));
      // use the rhs execution space for the copy, because otherwise we'd need a
      // fence on that execution space after the copy in any case, and this way
      // we can possibly avoid a fence before the copy
      K::deep_copy(
        rhs.state
          ->m_exec_spaces[
            rhs.state->next_pre_compute_exec_space_unlocked()].space,
        pool,
        rhs.pool);
      rhs.state->fence_unlocked();
      for (size_t i = 0; i < num_cf_groups; ++i) {
        // don't need cf_h, since the previous fence ensures that the copy from
        // host has completed
        cf_d[i] =
          cfd_view(
            pool.data() + (rhs.cf_d[i].data() - rhs.pool.data()),
            rhs.cf_d[i].layout());
      }
    }
    return *this;
  }

  CFPool&
  operator=(CFPool&& rhs) noexcept {

    if (state)
      state->fence_unlocked();
    if (rhs.state)
      rhs.state->fence_unlocked();
    std::swap(pool, rhs.pool);
    std::swap(num_cf_groups, rhs.num_cf_groups);
    std::swap(max_cf_extent_y, rhs.max_cf_extent_y);
    std::swap(cf_d, rhs.cf_d);
    std::swap(cf_h, rhs.cf_h);
    std::swap(cf_radii, rhs.cf_radii);
    return *this;
  }

  virtual ~CFPool() {
    reset();
  }

  static size_t
  cf_size(const CFArrayShape* cf, unsigned grp) {
    auto layout = impl::CFLayout<kokkos_device>::dimensions(cf, grp);
    // TODO: it would be best to use the following to compute
    // allocation size, but it is not implemented in Kokkos
    // 'auto alloc_sz = cfd_view::required_allocation_size(layout)'
    auto alloc_sz =
      impl::cf_view<typename kokkos_device::array_layout, memory_space>
      ::required_allocation_size(
        layout.dimension[0],
        layout.dimension[1],
        layout.dimension[2],
        layout.dimension[3],
        layout.dimension[4],
        layout.dimension[5]);
    return ((alloc_sz + (sizeof(impl::cf_t) - 1)) / sizeof(impl::cf_t));
  }

  static size_t
  pool_size(const CFArrayShape* cf) {
    size_t result = 0;
    if (cf)
      for (unsigned grp = 0; grp < cf->num_groups(); ++grp)
        result += cf_size(cf, grp);
    return result;
  }

  void
  prepare_pool(const CFArrayShape* cf, bool force = false) {
    auto current_pool_size = pool.extent(0);
    auto min_pool = pool_size(cf);
    reset((min_pool > current_pool_size) || force);
    if ((min_pool > current_pool_size) || (force && min_pool > 0))
      pool = decltype(pool)(K::ViewAllocateWithoutInitializing("cf"), min_pool);
  }

  void
  add_cf_group(
    const std::array<unsigned, 2>& radii,
    cfd_view cfd,
    std::any cfh) {

    assert(num_cf_groups < HPG_MAX_NUM_CF_GROUPS);
    cf_d[num_cf_groups] = cfd;
    cf_h.push_back(cfh);
    cf_radii[num_cf_groups] =
      {int(radii[0]), int(radii[1])};
    ++num_cf_groups;
    max_cf_extent_y =
      std::max(
        max_cf_extent_y,
        unsigned(cfd.extent(int(impl::core::CFAxis::y_major))));
  }

  void
  add_host_cfs(Device host_device, execution_space espace, CFArray&& cf_array) {
    prepare_pool(&cf_array);
    num_cf_groups = 0;
    size_t offset = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + offset,
        impl::CFLayout<kokkos_device>::dimensions(&cf_array, grp));
#ifndef NDEBUG
      std::cout << "alloc cf sz " << cf_init.extent(0)
                << " " << cf_init.extent(1)
                << " " << cf_init.extent(2)
                << " " << cf_init.extent(3)
                << " " << cf_init.extent(4)
                << " " << cf_init.extent(5)
                << std::endl;
      std::cout << "alloc cf str " << cf_init.stride(0)
                << " " << cf_init.stride(1)
                << " " << cf_init.stride(2)
                << " " << cf_init.stride(3)
                << " " << cf_init.stride(4)
                << " " << cf_init.stride(5)
                << std::endl;
#endif // NDEBUG

      typename decltype(cf_init)::HostMirror cf_h;
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial: {
        using host_device = impl::DeviceT<Device::Serial>::kokkos_device;
        cf_h = K::create_mirror_view(cf_init);
        impl::init_cf_host<host_device>(cf_h, cf_array, grp);
        K::deep_copy(espace, cf_init, cf_h);
        break;
      }
#endif // HPG_ENABLE_SERIAL
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP: {
        using host_device = impl::DeviceT<Device::OpenMP>::kokkos_device;
        cf_h = K::create_mirror_view(cf_init);
        impl::init_cf_host<host_device>(cf_h, cf_array, grp);
        K::deep_copy(espace, cf_init, cf_h);
        break;
      }
#endif // HPG_ENABLE_SERIAL
      default:
        assert(false);
        break;
      }
      offset += cf_size(&cf_array, grp);
      add_cf_group(cf_array.radii(grp), cf_init, cf_h);
    }
  }

  void
  add_device_cfs(
    execution_space espace,
    typename impl::DeviceCFArray<D>&& cf_array) {

    prepare_pool(&cf_array);
    num_cf_groups = 0;
    size_t offset = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + offset,
        impl::CFLayout<kokkos_device>::dimensions(&cf_array, grp));
#ifndef NDEBUG
      std::cout << "alloc cf sz " << cf_init.extent(0)
                << " " << cf_init.extent(1)
                << " " << cf_init.extent(2)
                << " " << cf_init.extent(3)
                << " " << cf_init.extent(4)
                << " " << cf_init.extent(5)
                << std::endl;
      std::cout << "alloc cf str " << cf_init.stride(0)
                << " " << cf_init.stride(1)
                << " " << cf_init.stride(2)
                << " " << cf_init.stride(3)
                << " " << cf_init.stride(4)
                << " " << cf_init.stride(5)
                << std::endl;
#endif // NDEBUG

      K::deep_copy(espace, cf_init, cf_array.m_views[grp]);
      offset += cf_size(&cf_array, grp);
      add_cf_group(
        cf_array.radii(grp),
        cf_init,
        std::make_tuple(
          std::move(cf_array.m_arrays[grp]),
          cf_array.m_views[grp]));
    }
  }

  void
  reset(bool free_pool = true) {
    if (state && pool.is_allocated()) {
      if (free_pool)
        pool = decltype(pool)();
      cf_h.clear();
      for (size_t i = 0; i < num_cf_groups; ++i)
        cf_d[i] = cfd_view();
      num_cf_groups = 0;
    }
  }
};

/** container for data and views associated with one stream of an execution
 * space*/
template <Device D>
struct /*HPG_EXPORT*/ ExecSpace final {
  using kokkos_device = typename impl::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;

  execution_space space;
  impl::visbuff_view<memory_space> visbuff;
  impl::gvisbuff_view<memory_space> gvisbuff;
  std::variant<
    impl::visdata_view<1, memory_space>,
    impl::visdata_view<2, memory_space>,
    impl::visdata_view<3, memory_space>,
    impl::visdata_view<4, memory_space>> visibilities;
  std::variant<
    impl::vector_view<impl::visdata_t<1>>,
    impl::vector_view<impl::visdata_t<2>>,
    impl::vector_view<impl::visdata_t<3>>,
    impl::vector_view<impl::visdata_t<4>>> visibilities_h;
  std::variant<
    std::vector<::hpg::VisData<1>>,
    std::vector<::hpg::VisData<2>>,
    std::vector<::hpg::VisData<3>>,
    std::vector<::hpg::VisData<4>>> vis_vector;
  mutable State::maybe_vis_t vis_promise;
  size_t num_visibilities;
  impl::weight_values_view<memory_space> weight_values;
  impl::vector_view<vis_weight_fp> weight_values_h;
  std::vector<vis_weight_fp> weight_values_vector;
  impl::weight_col_index_view<memory_space> weight_col_index;
  impl::vector_view<unsigned> weight_col_index_h;
  std::vector<unsigned> weight_col_index_vector;
  impl::weight_row_index_view<memory_space> weight_row_index;
  impl::vector_view<size_t> weight_row_index_h;
  std::vector<size_t> weight_row_index_vector;

  ExecSpace(execution_space sp)
    : space(sp) {
  }

  ExecSpace(const ExecSpace&) = delete;

  ExecSpace(ExecSpace&& other) noexcept
    : space(std::move(other).space)
    , visbuff(std::move(other).visbuff)
    , gvisbuff(std::move(other).gvisbuff)
    , visibilities(std::move(other).visibilities)
    , visibilities_h(std::move(other).visibilities_h)
    , vis_vector(std::move(other).vis_vector)
    , vis_promise(std::move(other).vis_promise)
    , num_visibilities(std::move(other).num_visibilities)
    , weight_values(std::move(other).weight_values)
    , weight_values_h(std::move(other).weight_values_h)
    , weight_values_vector(std::move(other).weight_values_vector)
    , weight_col_index(std::move(other).weight_col_index)
    , weight_col_index_h(std::move(other).weight_col_index_h)
    , weight_col_index_vector(std::move(other).weight_col_index_vector)
    , weight_row_index(std::move(other).weight_row_index)
    , weight_row_index_h(std::move(other).weight_row_index_h)
    , weight_row_index_vector(std::move(other).weight_row_index_vector) {
  }

  virtual ~ExecSpace() {}

  ExecSpace&
  operator=(ExecSpace&& rhs) noexcept {
    space = std::move(rhs).space;
    visbuff = std::move(rhs).visbuff;
    gvisbuff = std::move(rhs).gvisbuff;
    visibilities = std::move(rhs).visibilities;
    visibilities_h = std::move(rhs).visibilities_h;
    vis_vector = std::move(rhs).vis_vector;
    vis_promise = std::move(rhs).vis_promise;
    num_visibilities = std::move(rhs).num_visibilities;
    weight_values = std::move(rhs).weight_values;
    weight_values_h = std::move(rhs).weight_values_h;
    weight_values_vector = std::move(rhs).weight_values_vector;
    weight_col_index = std::move(rhs).weight_col_index;
    weight_col_index_h = std::move(rhs).weight_col_index_h;
    weight_col_index_vector = std::move(rhs).weight_col_index_vector;
    weight_row_index = std::move(rhs).weight_row_index;
    weight_row_index_h = std::move(rhs).weight_row_index_h;
    weight_row_index_vector = std::move(rhs).weight_row_index_vector;
    return *this;
  }

  template <unsigned N>
  constexpr impl::visdata_view<N, memory_space>
  visdata() {
    return std::get<impl::visdata_view<N, memory_space>>(visibilities);
  }

  template <unsigned N>
  size_t
  copy_visibilities_to_device(std::vector<::hpg::VisData<N>>&& in_vis) {

    num_visibilities = in_vis.size();
    if (num_visibilities > 0) {
      visibilities_h =
        impl::vector_view<impl::visdata_t<N>>(
          reinterpret_cast<impl::visdata_t<N>*>(in_vis.data()),
          num_visibilities);
      auto hview =
        std::get<impl::vector_view<impl::visdata_t<N>>>(
          visibilities_h);
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        visibilities =
          impl::visdata_view<N, memory_space>(
            reinterpret_cast<impl::visdata_t<N>*>(visbuff.data()),
            num_visibilities);
        auto dview =
          K::subview(visdata<N>(), std::pair((size_t)0, num_visibilities));
        K::deep_copy(space, dview, hview);
      } else {
        visibilities =
          impl::visdata_view<N, memory_space>(
            reinterpret_cast<impl::visdata_t<N>*>(&hview(0)),
            num_visibilities);
      }
    }
    vis_vector = std::move(in_vis);
    return num_visibilities;
  }

  std::tuple<size_t, size_t>
  copy_weights_to_device(
    std::vector<vis_weight_fp>&& in_weight_values,
    std::vector<unsigned>&& in_weight_col_index,
    std::vector<size_t>&& in_weight_row_index) {

    weight_values_vector = std::move(in_weight_values);
    weight_col_index_vector = std::move(in_weight_col_index);
    weight_row_index_vector = std::move(in_weight_row_index);

    if (weight_row_index_vector.size() > 0) {
      weight_values_h =
        impl::vector_view<vis_weight_fp>(
          weight_values_vector.data(),
          weight_values_vector.size());
      weight_col_index_h =
        impl::vector_view<unsigned>(
          weight_col_index_vector.data(),
          weight_col_index_vector.size());
      weight_row_index_h =
        impl::vector_view<size_t>(
          weight_row_index_vector.data(),
          weight_row_index_vector.size());
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        {
          auto weight_values_d =
            K::subview(
              weight_values,
              std::pair((size_t)0, weight_values_vector.size()));
          K::deep_copy(space, weight_values_d, weight_values_h);
        }
        {
          auto weight_col_index_d =
            K::subview(
              weight_col_index,
              std::pair((size_t)0, weight_col_index_vector.size()));
          K::deep_copy(space, weight_col_index_d, weight_col_index_h);
        }
        {
          auto weight_row_index_d =
            K::subview(
              weight_row_index,
              std::pair((size_t)0, weight_row_index_vector.size()));
          K::deep_copy(space, weight_row_index_d, weight_row_index_h);
        }
      } else {
        weight_values =
          impl::weight_values_view<memory_space>(
            weight_values_h.data(),
            weight_values_vector.size());
        weight_col_index =
          impl::weight_col_index_view<memory_space>(
            weight_col_index_h.data(),
            weight_col_index_vector.size());
        weight_row_index =
          impl::weight_row_index_view<memory_space>(
            weight_row_index_h.data(),
            weight_row_index_vector.size());
      }
    }
    return {weight_values_vector.size(), weight_row_index_vector.size()};
  }

  State::maybe_vis_t
  copy_visibilities_to_host(bool return_visibilities) {

    State::maybe_vis_t result;
    if (return_visibilities) {
      vis_promise =
        std::make_shared<std::shared_ptr<std::optional<VisDataVector>>>(
          std::make_shared<std::optional<VisDataVector>>(std::nullopt));
      result = vis_promise;
      std::visit(
        overloaded {
          [this](auto& v) {
            using v_t =
              std::remove_const_t<std::remove_reference_t<decltype(v)>>;
            if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
              constexpr unsigned N = v_t::value_type::npol;
              if (num_visibilities > 0) {
                auto hview =
                  std::get<impl::vector_view<impl::visdata_t<N>>>(
                    visibilities_h);
                auto dview =
                  K::subview(
                    visdata<N>(),
                    std::pair((size_t)0, num_visibilities));
                K::deep_copy(space, hview, dview);
              }
            }
          }
        },
        vis_vector);
    }
    return result;
  }

  void
  fence() noexcept {
    space.fence();
    if (vis_promise) {
      std::visit(
        overloaded {
          [this](auto& v) {
            using v_t =
              std::remove_const_t<std::remove_reference_t<decltype(v)>>;
            auto vdv =
              std::make_shared<std::optional<VisDataVector>>(
                VisDataVector(std::get<v_t>(std::move(vis_vector))));
            std::atomic_store(&*vis_promise, vdv);
          }
        },
        vis_vector);
      vis_promise.reset();
    }
  }
};

/** Kokkos state implementation for a device type */
template <Device D>
struct /*HPG_EXPORT*/ StateT
  : virtual public State {
public:

  Device m_device; /**< device type */
  unsigned m_max_active_tasks; /**< maximum number of active tasks */
  size_t m_visibility_batch_size; /**< number of visibilities to sent to
                                       gridding kernel at once */
  size_t m_max_avg_channels_per_vis; /**< max avg number of channel indexes for
                                          gridding */
  K::Array<int, 4> m_grid_size_global; /**< global grid size */
  K::Array<int, 4> m_grid_offset_local;/**< local grid offset*/
  K::Array<int, 4> m_grid_size_local; /**< local grid size */
  K::Array<grid_scale_fp, 2> m_grid_scale; /**< grid scale */
  unsigned m_num_polarizations; /**< number of visibility polarizations */
  std::array<unsigned, 4> m_implementation_versions; /**< impl versions*/

  using State::maybe_vis_t;

  using kokkos_device = typename impl::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using device_traits = DeviceTraits<kokkos_device>;
  using stream_type = typename device_traits::stream_type;
  using grid_layout =  impl::GridLayout<kokkos_device>;

  impl::grid_view<typename grid_layout::layout, memory_space> m_grid;
  impl::grid_weight_view<typename execution_space::array_layout, memory_space>
    m_grid_weights;
  impl::grid_view<typename grid_layout::layout, memory_space> m_model;
  impl::const_mindex_view<memory_space> m_mueller_indexes;
  impl::const_mindex_view<memory_space> m_conjugate_mueller_indexes;

  // use multiple execution spaces to support overlap of data copying with
  // computation when possible
  std::vector<std::conditional_t<std::is_void_v<stream_type>, int, stream_type>>
    m_streams;

protected:

  mutable std::mutex m_mtx;
  // access to the following members in const methods must be protected by m_mtx
  // (intentionally do not provide any thread safety guarantee outside of const
  // methods!)
  mutable std::vector<ExecSpace<D>> m_exec_spaces;
  mutable std::vector<std::tuple<CFPool<D>, std::optional<int>>> m_cfs;
  mutable std::deque<int> m_exec_space_indexes;
  mutable std::deque<int> m_cf_indexes;
  mutable StreamPhase m_current = StreamPhase::PRE_GRIDDING;

public:

  StateT(
    unsigned max_active_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size_global,
    const std::array<unsigned, 4>& grid_offset_local,
    const std::array<unsigned, 4>& grid_size_local,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
    : m_device(D)
    , m_max_active_tasks(
      std::min(max_active_tasks, device_traits::active_task_limit))
    , m_visibility_batch_size(visibility_batch_size)
    , m_max_avg_channels_per_vis(max_avg_channels_per_vis)
    , m_grid_scale({grid_scale[0], grid_scale[1]})
    , m_num_polarizations(mueller_indexes.m_npol)
    , m_implementation_versions(implementation_versions) {

    m_grid_size_global[int(impl::core::GridAxis::x)] =
      grid_size_global[GridValueArray::Axis::x];
    m_grid_size_global[int(impl::core::GridAxis::y)] =
      grid_size_global[GridValueArray::Axis::y];
    m_grid_size_global[int(impl::core::GridAxis::mrow)] =
      grid_size_global[GridValueArray::Axis::mrow];
    m_grid_size_global[int(impl::core::GridAxis::channel)] =
      grid_size_global[GridValueArray::Axis::channel];

    m_grid_offset_local[int(impl::core::GridAxis::x)] =
      grid_offset_local[GridValueArray::Axis::x];
    m_grid_offset_local[int(impl::core::GridAxis::y)] =
      grid_offset_local[GridValueArray::Axis::y];
    m_grid_offset_local[int(impl::core::GridAxis::mrow)] =
      grid_offset_local[GridValueArray::Axis::mrow];
    m_grid_offset_local[int(impl::core::GridAxis::channel)] =
      grid_offset_local[GridValueArray::Axis::channel];

    m_grid_size_local[int(impl::core::GridAxis::x)] =
      grid_size_local[GridValueArray::Axis::x];
    m_grid_size_local[int(impl::core::GridAxis::y)] =
      grid_size_local[GridValueArray::Axis::y];
    m_grid_size_local[int(impl::core::GridAxis::mrow)] =
      grid_size_local[GridValueArray::Axis::mrow];
    m_grid_size_local[int(impl::core::GridAxis::channel)] =
      grid_size_local[GridValueArray::Axis::channel];

    init_state(init_cf_shape);
    m_mueller_indexes =
      init_mueller("mueller_indexes", mueller_indexes);
    m_conjugate_mueller_indexes =
      init_mueller("conjugate_mueller_indexes", conjugate_mueller_indexes);
    new_grid(true, true);
  }

  StateT(const StateT& st)
    : m_device(D)
    , m_max_active_tasks(st.m_max_active_tasks)
    , m_visibility_batch_size(st.m_visibility_batch_size)
    , m_max_avg_channels_per_vis(st.m_max_avg_channels_per_vis)
    , m_grid_size_global(st.m_grid_size_global)
    , m_grid_offset_local(st.m_grid_offset_local)
    , m_grid_size_local(st.m_grid_size_local)
    , m_grid_scale(st.m_grid_scale)
    , m_num_polarizations(st.m_num_polarizations)
    , m_implementation_versions(st.m_implementation_versions) {

    std::scoped_lock lock(st.m_mtx);
    st.fence_unlocked();
    init_state(&st);
    m_mueller_indexes = st.m_mueller_indexes;
    m_conjugate_mueller_indexes = st.m_conjugate_mueller_indexes;
    new_grid(&st, true);
  }

  StateT(StateT&& st) noexcept
    : m_device(D) {

    m_max_active_tasks = std::move(st).m_max_active_tasks;
    m_visibility_batch_size = std::move(st).m_visibility_batch_size;
    m_max_avg_channels_per_vis = std::move(st).m_max_avg_channels_per_vis;
    m_grid_size_global = std::move(st).m_grid_size_global;
    m_grid_offset_local = std::move(st).m_grid_offset_local;
    m_grid_size_local = std::move(st).m_grid_size_local;
    m_grid_scale = std::move(st).m_grid_scale;
    m_num_polarizations = std::move(st).m_num_polarizations;
    m_implementation_versions = std::move(st).m_implementation_versions;

    m_grid = std::move(st).m_grid;
    m_grid_weights = std::move(st).m_grid_weights;
    m_model = std::move(st).m_model;
    m_mueller_indexes = std::move(st).m_mueller_indexes;
    m_conjugate_mueller_indexes = std::move(st).m_conjugate_mueller_indexes;
    m_streams = std::move(st).m_streams;
    m_exec_spaces = std::move(st).m_exec_spaces;
    m_exec_space_indexes = std::move(st).m_exec_space_indexes;
    m_current = std::move(st).m_current;

    m_cf_indexes = std::move(st).m_cf_indexes;
    m_cfs.resize(m_max_active_tasks);
    for (auto& [cf, last] : m_cfs)
      cf.state = this;
    for (size_t i = 0; i < m_max_active_tasks; ++i) {
      auto tmp_cf = std::move(m_cfs[i]);
      std::get<0>(tmp_cf).state = this;
      m_cfs[i] = std::move(st.m_cfs[i]);
      st.m_cfs[i] = std::move(tmp_cf);
    }
  }

  virtual ~StateT() {
    fence();
    m_grid = decltype(m_grid)();
    m_grid_weights = decltype(m_grid_weights)();
    m_model = decltype(m_model)();
    m_mueller_indexes = decltype(m_mueller_indexes)();
    m_conjugate_mueller_indexes = decltype(m_conjugate_mueller_indexes)();
    for (auto& [cf, last] : m_cfs)
      cf.reset();
    m_exec_spaces.clear();
    if constexpr(!std::is_void_v<stream_type>) {
      for (auto& str : m_streams) {
        auto rc = device_traits::destroy_stream(str);
        assert(rc);
      }
    }
  }

  StateT&
  operator=(const StateT& st) {
    StateT tmp(st);
    this->swap(tmp);
    return *this;
  }

  StateT&
  operator=(StateT&& st) noexcept {
    StateT tmp(std::move(st));
    this->swap(tmp);
    return *this;
  }

  StateT
  copy() const {
    return StateT(*this);
  }

  Device
  device() const noexcept override {
    return m_device;
  }

  unsigned
  visibility_gridder_version() const noexcept override {
    return m_implementation_versions[0];
  }

  unsigned
  grid_normalizer_version() const noexcept override {
    return m_implementation_versions[1];
  }

  unsigned
  fft_version() const noexcept override {
    return m_implementation_versions[2];
  }

  unsigned
  grid_shifter_version() const noexcept override {
    return m_implementation_versions[3];
  }

  unsigned
  max_active_tasks() const noexcept override {
    return m_max_active_tasks;
  }

  size_t
  visibility_batch_size() const noexcept override {
    return m_visibility_batch_size;
  }

  size_t
  max_avg_channels_per_vis() const noexcept override {
    return m_max_avg_channels_per_vis;
  }

  std::array<unsigned, 4>
  grid_size_global() const noexcept override {
    std::array<unsigned, 4> result;
    result[int(GridValueArray::Axis::x)] =
      m_grid_size_global[int(impl::core::GridAxis::x)];
    result[int(GridValueArray::Axis::y)]=
      m_grid_size_global[int(impl::core::GridAxis::y)];
    result[int(GridValueArray::Axis::mrow)]=
      m_grid_size_global[int(impl::core::GridAxis::mrow)];
    result[int(GridValueArray::Axis::channel)]=
      m_grid_size_global[int(impl::core::GridAxis::channel)];
    return result;
  }

  std::array<unsigned, 4>
  grid_size_local() const noexcept override {
    std::array<unsigned, 4> result;
    result[int(GridValueArray::Axis::x)] =
      m_grid_size_local[int(impl::core::GridAxis::x)];
    result[int(GridValueArray::Axis::y)]=
      m_grid_size_local[int(impl::core::GridAxis::y)];
    result[int(GridValueArray::Axis::mrow)]=
      m_grid_size_local[int(impl::core::GridAxis::mrow)];
    result[int(GridValueArray::Axis::channel)]=
      m_grid_size_local[int(impl::core::GridAxis::channel)];
    return result;
  }

  std::array<unsigned, 4>
  grid_offset_local() const noexcept override {
    std::array<unsigned, 4> result;
    result[int(GridValueArray::Axis::x)] =
      m_grid_offset_local[int(impl::core::GridAxis::x)];
    result[int(GridValueArray::Axis::y)]=
      m_grid_offset_local[int(impl::core::GridAxis::y)];
    result[int(GridValueArray::Axis::mrow)]=
      m_grid_offset_local[int(impl::core::GridAxis::mrow)];
    result[int(GridValueArray::Axis::channel)]=
      m_grid_offset_local[int(impl::core::GridAxis::channel)];
    return result;
  }

  std::array<grid_scale_fp, 2>
  grid_scale() const noexcept override {
    return {m_grid_scale[0], m_grid_scale[1]};
  }

  unsigned
  num_polarizations() const noexcept override {
    return m_num_polarizations;
  }

  virtual size_t
  convolution_function_region_size(const CFArrayShape* shape)
    const noexcept override {
    std::scoped_lock lock(m_mtx);
    return
      shape
      ? std::get<0>(m_cfs[0]).pool_size(shape)
      : std::get<0>(m_cfs[m_cf_indexes.front()]).pool.extent(0);
  }

  virtual std::optional<std::unique_ptr<Error>>
  allocate_convolution_function_region(
    unsigned context,
    const CFArrayShape* shape) override {
    return std::nullopt;
  }

  virtual std::optional<std::unique_ptr<Error>>
  set_convolution_function(
    unsigned context,
    Device host_device,
    CFArray&& cf_array) override {

    switch_cf_pool();
    auto& exec = m_exec_spaces[next_pre_compute_exec_space()];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    try {
      cf.add_device_cfs(
        exec.space,
        std::move(dynamic_cast<typename impl::DeviceCFArray<D>&&>(cf_array)));
    } catch (const std::bad_cast&) {
      cf.add_host_cfs(host_device, exec.space, std::move(cf_array));
    }
    return std::nullopt;
  }

  virtual std::optional<std::unique_ptr<Error>>
  set_model(Device host_device, GridValueArray&& gv) override {
    K::Array<int, 4> model_sz;
    model_sz[int(impl::core::GridAxis::x)] =
      gv.extent(unsigned(GridValueArray::Axis::x));
    model_sz[int(impl::core::GridAxis::y)] =
      gv.extent(unsigned(GridValueArray::Axis::y));
    model_sz[int(impl::core::GridAxis::mrow)] =
      gv.extent(unsigned(GridValueArray::Axis::mrow));
    model_sz[int(impl::core::GridAxis::channel)] =
      gv.extent(unsigned(GridValueArray::Axis::channel));
    if (m_grid_size_local[0] != model_sz[0]
        || m_grid_size_local[1] != model_sz[1]
        || m_grid_size_local[2] != model_sz[2]
        || m_grid_size_local[3] != model_sz[3])
      return
        std::make_unique<InvalidModelGridSizeError>(
          model_sz,
          m_grid_size_local);

    fence();
    auto& exec = m_exec_spaces[next_pre_compute_exec_space()];

    if (!m_model.is_allocated())
      m_model =
        decltype(m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          grid_layout::dimensions(m_grid_size_local));

    try {
      impl::GridValueViewArray<D> gvv =
        std::move(dynamic_cast<impl::GridValueViewArray<D>&&>(gv));
      K::deep_copy(exec.space, m_model, gvv.grid);
    } catch (const std::bad_cast&) {
      auto model_h = K::create_mirror_view(m_model);
      switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
      case Device::Serial:
        impl::init_model<Device::Serial>(model_h, gv);
        break;
#endif
#ifdef HPG_ENABLE_OPENMP
      case Device::OpenMP:
        impl::init_model<Device::OpenMP>(model_h, gv);
        break;
#endif
      default:
        return std::make_unique<DisabledHostDeviceError>();
        break;
      }
      K::deep_copy(exec.space, m_model, model_h);
    }
    return std::nullopt;
  }

  template <unsigned N>
  maybe_vis_t
  default_grid_visibilities(
    unsigned context,
    Device /*host_device*/,
    std::vector<::hpg::VisData<N>>&& visibilities,
    std::vector<vis_weight_fp>&& wgt_values,
    std::vector<unsigned>&& wgt_col_index,
    std::vector<size_t>&& wgt_row_index,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

    auto& exec_pre = m_exec_spaces[next_pre_compute_exec_space()];
    int len = exec_pre.copy_visibilities_to_device(std::move(visibilities));
    exec_pre.copy_weights_to_device(
      std::move(wgt_values),
      std::move(wgt_col_index),
      std::move(wgt_row_index));

    auto& exec_grid = m_exec_spaces[next_compute_exec_space()];
    auto& cf = std::get<0>(m_cfs[m_cf_indexes.front()]);
    auto& gvisbuff = exec_grid.gvisbuff;
    using gvis0 =
      typename std::remove_reference_t<decltype(gvisbuff)>::value_type;
    K::deep_copy(exec_grid.space, gvisbuff, gvis0());

    auto gridder =
      impl::core::VisibilityGridder(
        exec_grid.space,
        cf.cf_d,
        cf.cf_radii,
        cf.max_cf_extent_y,
        m_mueller_indexes,
        m_conjugate_mueller_indexes,
        len,
        0,
        1,
        exec_grid.template visdata<N>(),
        exec_grid.weight_values,
        exec_grid.weight_col_index,
        exec_grid.weight_row_index,
        gvisbuff,
        m_grid_scale,
        m_grid,
        m_grid_weights,
        m_model,
        m_grid_offset_local,
        m_grid_size_global);

    if (do_degrid) {
      gridder.degrid_all(len);
      if (do_grid)
        gridder.vis_copy_residual_and_rescale(len);
      else
        gridder.vis_copy_predicted(len);
    } else {
      gridder.vis_rescale(len);
    }

    if (do_grid) {
      if (update_grid_weights)
        gridder.grid_all(len);
      else
        gridder.grid_all_no_weights(len);
    }
    return exec_grid.copy_visibilities_to_host(return_visibilities);
  }

  template <unsigned N>
  maybe_vis_t
  grid_visibilities(
    unsigned context,
    Device host_device,
    std::vector<::hpg::VisData<N>>&& visibilities,
    std::vector<vis_weight_fp>&& wgt_values,
    std::vector<unsigned>&& wgt_col_index,
    std::vector<size_t>&& wgt_row_index,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

// #ifndef NDEBUG
//     for (auto& [channel, supp] : *cf_indexes) {
//       auto& cfpool = std::get<0>(m_cfs[m_cf_indexes.front()]);
//       if ((supp >= cfpool.num_cf_groups)
//           || (channel >= cfpool.cf_d[supp].extent_int(5)))
//         return OutOfBoundsCFIndexError({channel, supp});
//     }
// #endif // NDEBUG

    switch (visibility_gridder_version()) {
    case 0:
      return
        default_grid_visibilities(
          context,
          host_device,
          std::move(visibilities),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    case 1:
      return
        alt_grid_visibilities(
          host_device,
          std::move(visibilities),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    default:
      assert(false);
      std::abort();
      break;
    }
  }

  virtual std::variant<std::unique_ptr<Error>, maybe_vis_t>
  grid_visibilities(
    unsigned context,
    Device host_device,
    VisDataVector&& visibilities,
    std::vector<vis_weight_fp>&& wgt_values,
    std::vector<unsigned>&& wgt_col_index,
    std::vector<size_t>&& wgt_row_index,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid)
    override {

    switch (visibilities.m_npol) {
    case 1:
      return
        grid_visibilities(
          context,
          host_device,
          std::move(*visibilities.m_v1),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
        break;
    case 2:
      return
        grid_visibilities(
          context,
          host_device,
          std::move(*visibilities.m_v2),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    case 3:
      return
        grid_visibilities(
          context,
          host_device,
          std::move(*visibilities.m_v3),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    case 4:
      return
        grid_visibilities(
          context,
          host_device,
          std::move(*visibilities.m_v4),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    default:
      assert(false);
      return std::make_unique<Error>("Assertion violation");
      break;
    }
  }

  virtual void
  fence() const noexcept override {
    std::scoped_lock lock(m_mtx);
    fence_unlocked();
  }

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const override {
    std::scoped_lock lock(m_mtx);
    return grid_weights_unlocked();
  }

  virtual std::shared_ptr<GridWeightArray::value_type>
  grid_weights_ptr() const override {
    return
      std::make_shared<
        impl::GridWeightPtr<
          typename execution_space::array_layout,
          memory_space>>(m_grid_weights)->ptr();
  }

  virtual size_t
  grid_weights_span() const override {
    return m_grid_weights.span();
  }

  virtual std::unique_ptr<GridValueArray>
  grid_values() const override {
    std::scoped_lock lock(m_mtx);
    return grid_values_unlocked();
  }

  virtual std::shared_ptr<GridValueArray::value_type>
  grid_values_ptr() const override {
    return
      std::make_shared<
        impl::GridValuePtr<typename grid_layout::layout, memory_space>>(m_grid)
      ->ptr();
  }

  virtual size_t
  grid_values_span() const override {
    return m_grid.span();
  }

  virtual std::unique_ptr<GridValueArray>
  model_values() const override {
    std::scoped_lock lock(m_mtx);
    return model_values_unlocked();
  }

  virtual std::shared_ptr<GridValueArray::value_type>
  model_values_ptr() const override {
    return
      std::make_shared<
        impl::GridValuePtr<typename grid_layout::layout, memory_space>>(m_model)
      ->ptr();
  }

  virtual size_t
  model_values_span() const override {
    return m_model.span();
  }

  virtual void
  reset_grid() override {
    fence();
    new_grid(true, true);
  }

  virtual void
  fill_grid(const impl::gv_t& val) override {
    auto& exec = m_exec_spaces[next_pre_compute_exec_space_unlocked()];
    auto g_h = K::create_mirror_view(m_grid);
    K::deep_copy(g_h, impl::gv_t(0));
    K::deep_copy(exec.space, m_grid, g_h);
  };

  virtual void
  fill_grid_weights(const grid_value_fp& val) override {
    auto& exec = m_exec_spaces[next_pre_compute_exec_space_unlocked()];
    auto w_h = K::create_mirror_view(m_grid_weights);
    K::deep_copy(w_h, grid_value_fp(0));
    K::deep_copy(exec.space, m_grid_weights, w_h);
  };

  virtual void
  reset_model() override {
    fence();
    m_model = decltype(m_model)();
  }

  virtual void
  normalize_by_weights(grid_value_fp wfactor) override {
    impl::core::GridNormalizer(
      m_exec_spaces[next_compute_exec_space()].space,
      m_grid,
      m_grid_weights,
      wfactor)
      .normalize();
  }

  virtual std::optional<std::unique_ptr<Error>>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    std::optional<std::unique_ptr<Error>> err;
    if (in_place) {
      switch (fft_version()) {
      case 0:
        err =
          impl::FFT<execution_space>::in_place_kernel(
            m_exec_spaces[next_compute_exec_space()].space,
            sign,
            m_grid);
        break;
      default:
        assert(false);
        break;
      }
    } else {
      typename
        impl::grid_view<typename grid_layout::layout, memory_space>::const_type
        pre_grid = m_grid;
      new_grid(false, false);
      switch (fft_version()) {
      case 0:
        err =
          impl::FFT<execution_space>::out_of_place_kernel(
            m_exec_spaces[next_compute_exec_space()].space,
            sign,
            pre_grid,
            m_grid);
        break;
      default:
        assert(false);
        break;
      }
    }
    // apply normalization
    impl::core::GridNormalizer(
      m_exec_spaces[next_compute_exec_space()].space,
      m_grid,
      norm)
      .normalize();
    return err;
  }

  virtual std::optional<std::unique_ptr<Error>>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    std::optional<std::unique_ptr<Error>> err;
    if (m_model.is_allocated()){
      if (in_place) {
        switch (fft_version()) {
        case 0:
          err =
            impl::FFT<execution_space>::in_place_kernel(
              m_exec_spaces[next_compute_exec_space()].space,
              sign,
              m_model);
          break;
        default:
          assert(false);
          break;
        }
      } else {
        typename
          impl::grid_view<typename grid_layout::layout, memory_space>
          ::const_type pre_model = m_model;
        m_model =
          decltype(m_model)(
            K::ViewAllocateWithoutInitializing("grid"),
            grid_layout::dimensions(m_grid_size_local));
        switch (fft_version()) {
        case 0:
          err =
            impl::FFT<execution_space>::out_of_place_kernel(
              m_exec_spaces[next_compute_exec_space()].space,
              sign,
              pre_model,
              m_model);
          break;
        default:
          assert(false);
          break;
        }
      }
      // apply normalization
      impl::core::GridNormalizer(
        m_exec_spaces[next_compute_exec_space()].space,
        m_model,
        norm)
        .normalize();
    }
    return err;
  }

  virtual void
  shift_grid(ShiftDirection direction) override {
    impl::core::GridShifter(
      m_exec_spaces[next_compute_exec_space()].space,
      direction,
      m_grid)
      .shift();
  }

  virtual void
  shift_model(ShiftDirection direction) override {
    impl::core::GridShifter(
      m_exec_spaces[next_compute_exec_space()].space,
      direction,
      m_model)
      .shift();
  }

protected:

  void
  fence_unlocked() const noexcept {
    for (auto& i : m_exec_space_indexes) {
      auto& exec = m_exec_spaces[i];
      exec.fence();
    }
    next_pre_compute_exec_space_unlocked();
  }

  std::unique_ptr<GridWeightArray>
  grid_weights_unlocked() const {
    fence_unlocked();
    auto& exec = m_exec_spaces[next_pre_compute_exec_space_unlocked()];
    auto wgts_h = K::create_mirror(m_grid_weights);
    K::deep_copy(exec.space, wgts_h, m_grid_weights);
    exec.fence();
    return std::make_unique<impl::GridWeightViewArray<D>>(wgts_h);
  }

  std::unique_ptr<GridValueArray>
  grid_values_unlocked() const noexcept {
    fence_unlocked();
    auto& exec = m_exec_spaces[next_pre_compute_exec_space_unlocked()];
    auto grid_h = K::create_mirror(m_grid);
    K::deep_copy(exec.space, grid_h, m_grid);
    exec.fence();
    return std::make_unique<impl::GridValueViewArray<D>>(grid_h);
  }

  std::unique_ptr<GridValueArray>
  model_values_unlocked() const {
    fence_unlocked();
    auto& exec = m_exec_spaces[next_pre_compute_exec_space_unlocked()];
    if (m_model.is_allocated()) {
      auto model_h = K::create_mirror(m_model);
      K::deep_copy(exec.space, model_h, m_model);
      exec.fence();
      return std::make_unique<impl::GridValueViewArray<D>>(model_h);
    } else {
      std::array<unsigned, 4> ex{
        unsigned(m_grid.extent(0)),
        unsigned(m_grid.extent(1)),
        unsigned(m_grid.extent(2)),
        unsigned(m_grid.extent(3))};
      return std::make_unique<impl::UnallocatedModelValueArray>(ex);
    }
  }

  void
  swap(StateT& other) noexcept {
    assert(m_max_active_tasks == other.m_max_active_tasks);
    std::swap(m_visibility_batch_size, other.m_visibility_batch_size);
    std::swap(
      m_max_avg_channels_per_vis,
      other.m_max_avg_channels_per_vis);
    std::swap(m_grid_size_global, other.m_grid_size_global);
    std::swap(m_grid_size_local, other.m_grid_size_local);
    std::swap(m_grid_offset_local, other.m_grid_offset_local);
    std::swap(m_grid_scale, other.m_grid_scale);
    std::swap(m_implementation_versions, other.m_implementation_versions);

    std::swap(m_grid, other.m_grid);
    std::swap(m_grid_weights, other.m_grid_weights);
    std::swap(m_model, other.m_model);
    std::swap(m_mueller_indexes, other.m_mueller_indexes);
    std::swap(m_conjugate_mueller_indexes, other.m_conjugate_mueller_indexes);
    std::swap(m_streams, other.m_streams);
    std::swap(m_exec_spaces, other.m_exec_spaces);
    std::swap(m_exec_space_indexes, other.m_exec_space_indexes);
    std::swap(m_current, other.m_current);

    for (size_t i = 0; i < m_max_active_tasks; ++i) {
      auto tmp_cf = std::move(m_cfs[i]);
      std::get<0>(tmp_cf).state = this;
      m_cfs[i] = std::move(other.m_cfs[i]);
      other.m_cfs[i] = std::move(tmp_cf);
    }
    std::swap(m_cf_indexes, other.m_cf_indexes);
  }

  void
  init_state(const std::variant<const CFArrayShape*, const StateT*>& init) {
    m_streams.resize(m_max_active_tasks);
    m_cfs.resize(m_max_active_tasks);
    for (unsigned i = 0; i < m_max_active_tasks; ++i) {
      if constexpr (!std::is_void_v<stream_type>) {
        auto rc = device_traits::create_stream(m_streams[i]);
        assert(rc);
        m_exec_spaces.emplace_back(execution_space(m_streams[i]));
        if (std::holds_alternative<const CFArrayShape*>(init)) {
          m_exec_space_indexes.push_back(i);
          m_cf_indexes.push_back(i);
        }
      } else {
        m_exec_spaces.emplace_back(execution_space());
        if (std::holds_alternative<const CFArrayShape*>(init)) {
          m_exec_space_indexes.push_back(i);
          m_cf_indexes.push_back(i);
        }
      }
      auto& esp = m_exec_spaces.back();
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        esp.visbuff =
          decltype(esp.visbuff)(
            K::ViewAllocateWithoutInitializing("visibility_buffer"),
            m_visibility_batch_size);
        auto max_num_channels =
          m_max_avg_channels_per_vis * m_visibility_batch_size;
        esp.weight_values =
          decltype(esp.weight_values)(
            K::ViewAllocateWithoutInitializing("weight_values"),
            max_num_channels);
        esp.weight_col_index =
          decltype(esp.weight_col_index)(
            K::ViewAllocateWithoutInitializing("weight_col_index"),
            max_num_channels);
        esp.weight_row_index =
          decltype(esp.weight_row_index)(
            K::ViewAllocateWithoutInitializing("weight_row_index"),
            m_visibility_batch_size + 1);
      }
      esp.gvisbuff =
        decltype(esp.gvisbuff)(
          K::ViewAllocateWithoutInitializing("gvis_buffer"),
          m_visibility_batch_size);
    }

    if (std::holds_alternative<const CFArrayShape*>(init)) {
      const CFArrayShape* init_cf_shape = std::get<const CFArrayShape*>(init);
      for (auto& [cf, last] : m_cfs) {
        cf.state = this;
        cf.prepare_pool(init_cf_shape, true);
        last.reset();
      }
    } else {
      const StateT* ost = std::get<const StateT*>(init);
      for (auto& i : ost->m_exec_space_indexes) {
        auto& esp = m_exec_spaces[i];
        m_exec_space_indexes.push_back(i);
        auto& st_esp = ost->m_exec_spaces[i];
        K::deep_copy(esp.space, esp.visbuff, st_esp.visbuff);
        K::deep_copy(esp.space, esp.gvisbuff, st_esp.gvisbuff);
        if constexpr (std::is_same_v<K::HostSpace, memory_space>) {
          std::visit(
            overloaded {
              [&esp](auto& v) {
                esp.visibilities = v;
              }
            },
            st_esp.visibilities);
        } else {
          std::visit(
            overloaded {
              [&esp](auto& v) {
                using v_t = std::remove_reference_t<decltype(v)>;
                constexpr unsigned N = v_t::value_type::npol;
                esp.visibilities =
                  impl::visdata_view<N, memory_space>(
                    reinterpret_cast<impl::visdata_t<N>*>(esp.visbuff.data()),
                    v.extent(0));
              }
            },
            st_esp.visibilities);
        }
        // FIXME: esp.copy_state = st_esp.copy_state;
      }
      for (auto& i : ost->m_cf_indexes) {
        auto& [cf, last] = m_cfs[i];
        m_cf_indexes.push_back(i);
        cf = std::get<0>(ost->m_cfs[i]);
        cf.state = this;
      }
      if (ost->m_model.is_allocated()) {
        m_model =
          decltype(m_model)(
            K::ViewAllocateWithoutInitializing("model"),
            grid_layout::dimensions(m_grid_size_local));
        K::deep_copy(m_exec_spaces[0].space, m_model, ost->m_model);
      }
    }
    next_pre_compute_exec_space_unlocked();
  }

  /** copy Mueller indexes to device */
  template <size_t N>
  impl::mindex_view<memory_space>
  copy_mueller_indexes(
    const std::string& name,
    const std::vector<iarray<N>>& mindexes) {

    auto& esp = m_exec_spaces[next_pre_compute_exec_space()];
    impl::mindex_view<memory_space> result(name);
    auto mueller_indexes_h = K::create_mirror_view(result);
    size_t mr = 0;
    for (; mr < mindexes.size(); ++mr) {
      auto& mi_row = mindexes[mr];
      size_t mc = 0;
      for (; mc < N; ++mc)
        mueller_indexes_h(mr, mc) = int(mi_row[mc]);
      for (; mc < result.extent(1); ++mc)
        mueller_indexes_h(mr, mc) = -1;
    }
    for (; mr < result.extent(0); ++mr)
      for (size_t mc = 0; mc < result.extent(1); ++mc)
        mueller_indexes_h(mr, mc) = -1;
    K::deep_copy(esp.space, result, mueller_indexes_h);
    return result;
  }

  impl::mindex_view<memory_space>
  init_mueller(const std::string& name, const IArrayVector& mueller_indexes) {

    switch (mueller_indexes.m_npol) {
    case 1:
      return copy_mueller_indexes(name, *mueller_indexes.m_v1);
      break;
    case 2:
      return copy_mueller_indexes(name, *mueller_indexes.m_v2);
      break;
    case 3:
      return copy_mueller_indexes(name, *mueller_indexes.m_v3);
      break;
    case 4:
      return copy_mueller_indexes(name, *mueller_indexes.m_v4);
      break;
    default:
      assert(false);
      return impl::mindex_view<memory_space>(name);
      break;
    }
  }

protected:

  friend class CFPool<D>;

  int
  next_exec_space_unlocked(StreamPhase next) const {
    int old_idx = m_exec_space_indexes.front();
    int new_idx = old_idx;
    if ((m_current == StreamPhase::DEGRIDDING
         || m_current == StreamPhase::GRIDDING)
        && (next == StreamPhase::PRE_GRIDDING
            || next == StreamPhase::PRE_DEGRIDDING)) {
      if (m_max_active_tasks > 1) {
        m_exec_space_indexes.push_back(old_idx);
        m_exec_space_indexes.pop_front();
        new_idx = m_exec_space_indexes.front();
      }
      m_exec_spaces[new_idx].fence();
      std::get<1>(m_cfs[m_cf_indexes.front()]) = new_idx;
    }
#ifndef NDEBUG
    std::cout << m_current << "(" << old_idx << ")->"
              << next << "(" << new_idx << ")"
              << std::endl;
#endif // NDEBUG
    m_current = next;
    return new_idx;
  }

  int
  next_exec_space(StreamPhase next) const {
    std::scoped_lock lock(m_mtx);
    return next_exec_space_unlocked(next);
  }

  virtual int
  next_pre_compute_exec_space_unlocked() const {
    return next_exec_space_unlocked(StreamPhase::PRE_GRIDDING);
  }

  int
  next_pre_compute_exec_space() const {
    std::scoped_lock lock(m_mtx);
    return next_pre_compute_exec_space_unlocked();
  }

  virtual int
  next_compute_exec_space_unlocked() const {
    return next_exec_space_unlocked(StreamPhase::GRIDDING);
  }

  int
  next_compute_exec_space() const {
    std::scoped_lock lock(m_mtx);
    return next_compute_exec_space_unlocked();
  }

  void
  switch_cf_pool() {
    auto esp_idx = next_pre_compute_exec_space();
    m_cf_indexes.push_back(m_cf_indexes.front());
    m_cf_indexes.pop_front();
    auto& [cf, last] = m_cfs[m_cf_indexes.front()];
    if (last.value_or(esp_idx) != esp_idx)
      m_exec_spaces[last.value()].fence();
    last = esp_idx;
  }

protected:

  void
  new_grid(std::variant<const StateT*, bool> source, bool also_weights) {

    const bool create_without_init =
      std::holds_alternative<const StateT*>(source) || !std::get<bool>(source);

    // in the following, we don't call next_exec_space() except when a stream is
    // required, as there are code paths that never use a stream, and thus we
    // can avoid unnecessary stream switches
    if (create_without_init)
      m_grid =
        decltype(m_grid)(
          K::ViewAllocateWithoutInitializing("grid"),
          grid_layout::dimensions(m_grid_size_local));
    else
      m_grid =
        decltype(m_grid)(
          K::view_alloc(
            "grid",
            m_exec_spaces[next_pre_compute_exec_space()].space),
          grid_layout::dimensions(m_grid_size_local));
#ifndef NDEBUG
    std::cout << "alloc grid sz " << m_grid.extent(0)
              << " " << m_grid.extent(1)
              << " " << m_grid.extent(2)
              << " " << m_grid.extent(3)
              << std::endl;
    std::cout << "alloc grid str " << m_grid.stride(0)
              << " " << m_grid.stride(1)
              << " " << m_grid.stride(2)
              << " " << m_grid.stride(3)
              << std::endl;
#endif // NDEBUG

    static_assert(
      int(impl::core::GridWeightAxis::mrow) == 0
      && int(impl::core::GridWeightAxis::channel) == 1);
    if (also_weights) {
      if (create_without_init)
        m_grid_weights =
          decltype(m_grid_weights)(
            K::ViewAllocateWithoutInitializing("grid_weights"),
            int(m_grid_size_local[int(impl::core::GridAxis::mrow)]),
            int(m_grid_size_local[int(impl::core::GridAxis::channel)]));
      else
        m_grid_weights =
          decltype(m_grid_weights)(
            K::view_alloc(
              "grid_weights",
              m_exec_spaces[next_pre_compute_exec_space()].space),
            int(m_grid_size_local[int(impl::core::GridAxis::mrow)]),
            int(m_grid_size_local[int(impl::core::GridAxis::channel)]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto st = std::get<const StateT*>(source);
      auto& exec = m_exec_spaces[next_pre_compute_exec_space()];
      K::deep_copy(exec.space, m_grid, st->m_grid);
      if (also_weights)
        K::deep_copy(exec.space, m_grid_weights, st->m_grid_weights);
    }
  }
};

/** helper class for calling methods of StateT member of GridderState
 * instances
 *
 * Manages calling the appropriate methods of StateT as well as updating the
 * StateT member */
struct /*HPG_EXPORT*/ GridderState {

  template <typename GS>
  static std::variant<std::unique_ptr<Error>, ::hpg::GridderState>
  allocate_convolution_function_region(GS&& st, const CFArrayShape* shape) {

    ::hpg::GridderState result(std::forward<GS>(st));
    if (auto error =
        result.m_impl->allocate_convolution_function_region(0, shape);
        error)
      return std::move(error.value());
    else
      return std::move(result);
  }

  template <typename GS>
  static std::variant<std::unique_ptr<Error>, ::hpg::GridderState>
  set_convolution_function(GS&& st, Device host_device, CFArray&& cf) {

    if (host_devices().count(host_device) > 0) {
      ::hpg::GridderState result(std::forward<GS>(st));
      if (auto error =
          result.m_impl
            ->set_convolution_function(0, host_device, std::move(cf));
          error)
        return std::move(error.value());
      else
        return std::move(result);
    } else {
      return std::make_unique<DisabledHostDeviceError>();
    }
  }

  template <typename GS>
  static std::variant<std::unique_ptr<Error>, ::hpg::GridderState>
  set_model(GS&& st, Device host_device, GridValueArray&& gv) {

    if (host_devices().count(host_device) > 0) {
      ::hpg::GridderState result(std::forward<GS>(st));
      if (auto error = result.m_impl->set_model(host_device, std::move(gv));
          error)
        return std::move(error.value());
      else
        return std::move(result);
    } else {
      return std::make_unique<DisabledHostDeviceError>();
    }
  }

  template <typename GS>
  static std::variant<
    std::unique_ptr<Error>,
    std::tuple<::hpg::GridderState, future<VisDataVector>>>
  grid_visibilities(
    GS&& st,
    Device host_device,
    VisDataVector&& visibilities,
    const std::vector<std::map<unsigned, vis_weight_fp>>& grid_channel_maps,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

    if (host_devices().count(host_device) == 0)
      return std::make_unique<DisabledHostDeviceError>();

    auto num_visibilities = visibilities.size();
    if (num_visibilities > st.m_impl->visibility_batch_size())
      return std::make_unique<ExcessiveNumberVisibilitiesError>();

    if (visibilities.m_npol != st.m_impl->num_polarizations())
      return std::make_unique<InvalidNumberPolarizationsError>();

    if (!do_grid && update_grid_weights)
      return std::make_unique<UpdateWeightsWithoutGriddingError>();

    if (num_visibilities != grid_channel_maps.size())
      return std::make_unique<GridChannelMapsSizeError>();

    // convert channel map to matrix in CRS format
    size_t max_num_channels =
      st.m_impl->max_avg_channels_per_vis() * num_visibilities;
    std::vector<vis_weight_fp> wgt;
    wgt.reserve(max_num_channels);
    std::vector<unsigned> col_index;
    col_index.reserve(max_num_channels);
    std::vector<size_t> row_index;
    row_index.reserve(num_visibilities + 1);
    for (size_t r = 0; r < num_visibilities; ++r) {
      row_index.push_back(col_index.size());
      for (auto& [c, w] : grid_channel_maps[r]) {
        col_index.push_back(c);
        wgt.push_back(w);
      }
    }
    if (col_index.size() > max_num_channels)
      return std::make_unique<ExcessiveVisibilityChannelsError>();
    row_index.push_back(col_index.size());

    ::hpg::GridderState result(std::forward<GS>(st));
    auto err_or_maybevis =
      result.m_impl->grid_visibilities(
        0,
        host_device,
        std::move(visibilities),
        std::move(wgt),
        std::move(col_index),
        std::move(row_index),
        update_grid_weights,
        do_degrid,
        return_visibilities,
        do_grid);
    if (std::holds_alternative<std::unique_ptr<Error>>(err_or_maybevis)) {
      return std::get<std::unique_ptr<Error>>(std::move(err_or_maybevis));
    } else {
      auto mvs = std::get<State::maybe_vis_t>(std::move(err_or_maybevis));
#if HPG_API >= 17
      return
        std::make_tuple(
          std::move(result),
          future<VisDataVector>(
            [mvs, result=opt_t<VisDataVector>()]() mutable
            -> opt_t<VisDataVector>& {
              if (!result) {
                if (mvs) {
                  auto mvs0 = std::atomic_load(&mvs);
                  if (*mvs0 && (*mvs0)->has_value())
                    result = std::move(**mvs0);
                }
              }
              return result;
            }));
#else
      return
        std::make_tuple(
          std::move(result),
          future<VisDataVector>(
            [mvs, result=opt_t<VisDataVector>()]() mutable
            -> opt_t<VisDataVector>& {
              if (!result) {
                if (mvs) {
                  auto mvs0 = std::atomic_load(&mvs);
                  if (*mvs0 && (*mvs0)->has_value())
                    result =
                      opt_t<VisDataVector>(
                        new VisDataVector(std::move(*mvs0)->value()));
                }
              }
              return result;
            }));
#endif
    }
  }

  template <typename GS>
  static std::variant<std::unique_ptr<Error>, ::hpg::GridderState>
  apply_grid_fft(GS&& st, grid_value_fp norm, FFTSign sign, bool in_place) {

    ::hpg::GridderState result(std::forward<GS>(st));
    if (auto error = result.m_impl->apply_grid_fft(norm, sign, in_place); error)
      return std::move(error.value());
    else
      return std::move(result);
  }

  template <typename GS>
  static std::variant<std::unique_ptr<Error>, ::hpg::GridderState>
  apply_model_fft(GS&& st, grid_value_fp norm, FFTSign sign, bool in_place) {

    ::hpg::GridderState result(std::forward<GS>(st));
    if (auto error = result.m_impl->apply_model_fft(norm, sign, in_place);
        error)
      return std::move(error.value());
    else
      return std::move(result);
  }
};

} // end namespace hpg::runtime

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
