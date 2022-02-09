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
  num_exec_contexts() const noexcept = 0;

  virtual unsigned
  visibility_gridder_version() const noexcept = 0;

  virtual unsigned
  grid_normalizer_version() const noexcept = 0;

  virtual unsigned
  fft_version() const noexcept = 0;

  virtual unsigned
  grid_shifter_version() const noexcept = 0;

  virtual unsigned
  num_active_tasks() const noexcept = 0;

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
  convolution_function_region_size(const CFArrayShape& shape)
    const noexcept = 0;

  virtual rval_t<size_t>
  current_convolution_function_region_size(unsigned context)
    const noexcept = 0;

  virtual std::optional<std::unique_ptr<Error>>
  allocate_convolution_function_region(const CFArrayShape& shape) = 0;

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
  COPY,
  COMPUTE,
};

/** formatted output for StreamPhase value */
std::ostream&
operator<<(std::ostream& str, const StreamPhase& ph);

struct DevCFShape
  : public CFArrayShape {

  std::vector<unsigned> m_shape;

  DevCFShape(const std::vector<unsigned>& shape)
    : m_shape(shape) {
    assert((m_shape.size() == 0) || ((m_shape.size() - 1) % (rank - 1) == 0));
  }

  DevCFShape(size_t num_grp) {
    m_shape.resize(1 + num_grp * (rank - 1));
  }

  DevCFShape() {}

  unsigned
  oversampling() const override {
    return (m_shape.size() > 0) ? m_shape[0] : 0;
  }

  unsigned
  num_groups() const override {
    return
      (m_shape.size() > 0) ? (unsigned((m_shape.size() - 1) / (rank - 1))) : 0;
  }

  std::array<unsigned, rank - 1>
  extents(unsigned grp) const override {
    std::array<unsigned, rank - 1> result;
    for (unsigned d = 0; d < rank - 1; ++d)
      result[d] = m_shape[grp * (rank - 1) + d + 1];
    return result;
  }
};

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

  K::View<impl::cf_t*, memory_space> pool;
  unsigned num_cf_groups;
  unsigned max_cf_extent_y;
  K::Array<cfd_view, HPG_MAX_NUM_CF_GROUPS> cf_d; // unmanaged (in pool)
  std::vector<std::any> cf_h;
  K::Array<K::Array<int, 2>, HPG_MAX_NUM_CF_GROUPS> cf_radii;
  size_t size;

  CFPool()
    : num_cf_groups(0)
    , max_cf_extent_y(0) {

    for (size_t i = 0; i < HPG_MAX_NUM_CF_GROUPS; ++i)
      cf_d[i] = cfd_view();
  }

  CFPool(const CFPool& other) = delete;

  CFPool(CFPool&& other) = delete;

  CFPool&
  operator=(const CFPool& rhs) = delete;

  CFPool&
  operator=(CFPool&& rhs) = delete;

  virtual ~CFPool() {
    reset();
  }

  DevCFShape
  shape() const {
    std::vector<unsigned> sh;
    sh.reserve(num_cf_groups * (CFArrayShape::rank - 1) + 1);
    if (num_cf_groups > 0) {
      sh.push_back(cf_d[0].extent(int(impl::core::CFAxis::x_minor)));
      for (unsigned i = 0; i < num_cf_groups; ++i) {
        sh.push_back(cf_d[i].extent(int(impl::core::CFAxis::x_major)) * sh[0]);
        sh.push_back(cf_d[i].extent(int(impl::core::CFAxis::y_major)) * sh[0]);
        sh.push_back(cf_d[i].extent(int(impl::core::CFAxis::mueller)));
        sh.push_back(cf_d[i].extent(int(impl::core::CFAxis::channel)));
      }
    }
    return DevCFShape(sh);
  }

  void
  copy_from(execution_space espace, const CFPool& other) {
    // caller must ensure that it's safe to overwrite array values in this
    // instance, and to read array values from 'other'
    num_cf_groups = other.num_cf_groups;
    max_cf_extent_y = other.max_cf_extent_y;
    size = other.size;
    cf_h.clear();
    cf_radii = other.cf_radii;
    if (pool.is_allocated())
      K::resize(pool, other.pool.extent(0));
    else
      pool =
        decltype(pool)(
          K::ViewAllocateWithoutInitializing("cf"),
          other.pool.extent(0));
    K::deep_copy(espace, pool, other.pool); // TODO: copy only "size" elements
    for (unsigned grp = 0; grp < num_cf_groups; ++grp)
      cf_d[grp] =
        cfd_view(
          pool.data() + (other.cf_d[grp].data() - other.pool.data()),
          other.cf_d[grp].layout());
  }

  static size_t
  cf_size(const CFArrayShape& cf, unsigned grp) {
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
  pool_size(const CFArrayShape& cf) {
    size_t result = 0;
    for (unsigned grp = 0; grp < cf.num_groups(); ++grp)
      result += cf_size(cf, grp);
    return result;
  }

  void
  prepare_pool(const CFArrayShape& cf, bool force = false) {
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
    std::optional<std::any> cfh) {

    assert(num_cf_groups < HPG_MAX_NUM_CF_GROUPS);
    cf_d[num_cf_groups] = cfd;
    if (cfh)
      cf_h.push_back(cfh.value());
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
    prepare_pool(cf_array);
    num_cf_groups = 0;
    size = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + size,
        impl::CFLayout<kokkos_device>::dimensions(cf_array, grp));
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
      size += cf_size(cf_array, grp);
      add_cf_group(cf_array.radii(grp), cf_init, cf_h);
    }
  }

  void
  add_device_cfs(
    execution_space espace,
    typename impl::DeviceCFArray<D>&& cf_array) {

    prepare_pool(cf_array);
    num_cf_groups = 0;
    size = 0;
    for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + size,
        impl::CFLayout<kokkos_device>::dimensions(cf_array, grp));
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
      size += cf_size(cf_array, grp);
      add_cf_group(
        cf_array.radii(grp),
        cf_init,
        std::make_tuple(
          std::move(cf_array.m_arrays[grp]),
          cf_array.m_views[grp]));
    }
  }

  void
  set_shape(const DevCFShape& shape) {
    prepare_pool(shape);
    num_cf_groups = 0;
    size = 0;
    for (unsigned grp = 0; grp < shape.num_groups(); ++grp) {
      cfd_view cf_init(
        pool.data() + size,
        impl::CFLayout<kokkos_device>::dimensions(shape, grp));
      size += cf_size(shape, grp);
      add_cf_group(shape.radii(grp), cf_init, std::nullopt);
    }
  }

  void
  reset(bool free_pool = true) {
    if (pool.is_allocated()) {
      if (free_pool)
        pool = decltype(pool)();
      cf_h.clear();
      for (size_t i = 0; i < num_cf_groups; ++i)
        cf_d[i] = cfd_view();
      num_cf_groups = 0;
    }
  }
};

/** container for all CFPool instances managed by a StateT instance */
template <Device D>
struct CFPoolRepo {
public:

  using pool_type = std::shared_ptr<CFPool<D>>;

  using id_type = unsigned;

private:

  std::set<pool_type> m_pools;

  mutable std::map<id_type, std::weak_ptr<CFPool<D>>> m_pool_ids;

  mutable std::recursive_mutex m_mtx;

  mutable id_type m_next_id;

public:

  CFPoolRepo() {}

  CFPoolRepo(size_t n) {
    std::generate_n(
      std::inserter(m_pools, m_pools.end()),
      n,
      [] { return std::make_shared<CFPool<D>>(); });
  }

  CFPoolRepo(const CFPoolRepo&) = delete;

  CFPoolRepo(CFPoolRepo&& other) noexcept
    : m_pools(std::move(other).m_pools)
    , m_pool_ids(std::move(other).m_pool_ids)
    , m_next_id(std::move(other).m_next_id) {}

  CFPoolRepo&
  operator=(const CFPoolRepo&) = delete;

  CFPoolRepo&
  operator=(const CFPoolRepo&& rhs) noexcept {
    std::scoped_lock lck(m_mtx);
    m_pools = std::move(rhs).m_pools;
    m_pool_ids = std::move(rhs).m_pool_ids;
    m_next_id = std::move(rhs).m_next_id;
    return *this;
  }

  std::map<const CFPool<D>*, std::weak_ptr<CFPool<D>>>
  copy_from(
    typename CFPool<D>::execution_space espace,
    const CFPoolRepo<D>& other) {

    std::scoped_lock lck(m_mtx, other.m_mtx);
    // caller must ensure that it's safe to overwrite values in this instance,
    // and to read values from 'other'...basically fence all streams using the
    // two instances prior to calling this method

    while (m_pools.size() > other.size())
      m_pools.erase(m_pools.end());
    while (m_pools.size() < other.size())
      m_pools.insert(std::make_shared<CFPool<D>>());
    m_pool_ids.clear();
    std::map<const CFPool<D>*, std::weak_ptr<CFPool<D>>> result;
    for (auto dst = m_pools.begin(), src = other.m_pools.begin();
         dst != m_pools.end();
         ++dst, ++src) {
      (*dst)->copy_from(espace, **src);
      auto maybe_id = other.get_id(*src);
      if (maybe_id)
        m_pool_ids[maybe_id.value()] = *dst;
      result[(*src).get()] = *dst;
    }
    m_next_id = other.m_next_id;
    // NB: the return value holds bare pointers to CFPool instances in 'other'
    // to avoid changing the reference count of those instances
    return result;
  }

  pool_type
  get_unused_pool(const std::optional<id_type>& id = std::nullopt) noexcept {
    std::scoped_lock lck(m_mtx);
    auto result =
      std::find_if(
        m_pools.begin(),
        m_pools.end(),
        [](const auto& p) { return p.use_count() == 1; });
    if (result != m_pools.end()) {
      if (id)
        set_id(*result, id.value());
      else
        set_id(*result, m_next_id++);
      return *result;
    }
    return pool_type();
  }

  size_t
  size() const {
    return m_pools.size();
  }

  template <typename F>
  void
  for_all_cf_pools(F&& f) {
    std::scoped_lock lck(m_mtx);
    for (auto& pool: m_pools)
      f(pool);
  }

  pool_type
  find_pool(id_type id) const {
    std::scoped_lock lck(m_mtx);
    if (m_pool_ids.count(id) > 0)
      return m_pool_ids.at(id).lock();
    return nullptr;
  }

  void
  set_id(const pool_type& p, id_type id) {
    std::scoped_lock lck(m_mtx);
    if (m_pools.count(p) > 0) {
      auto pid =
        std::find_if(
          m_pool_ids.begin(),
          m_pool_ids.end(),
          [&p](auto& i_p) {
            return std::get<1>(i_p).lock() == p;
          });
      if (pid != m_pool_ids.end())
        m_pool_ids.erase(pid);
      m_pool_ids[id] = p;
    }
  }

  std::optional<id_type>
  get_id(const pool_type& p) const {
    std::scoped_lock lck(m_mtx);
    auto ip =
      std::find_if(
        m_pool_ids.begin(),
        m_pool_ids.end(),
        [&p](auto& i_p) {
          return std::get<1>(i_p).lock() == p;
        });
    return
      (ip != m_pool_ids.end()) ? std::make_optional(ip->first) : std::nullopt;
  }
};

template <Device D, typename T, typename S=T>
struct StreamVector {
  using kokkos_device = typename impl::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using memory_traits =
    std::conditional_t<
      std::is_same_v<memory_space, K::HostSpace>,
      K::MemoryUnmanaged,
      K::MemoryManaged>;

  // TODO: improve the test for binary compatibility of T and S
  static_assert(sizeof(T) == sizeof(S));

  size_t m_max_size;
  K::View<T*, memory_space, memory_traits> m_values_d;
  K::View<T*, memory_space, K::MemoryUnmanaged> m_values;
  std::vector<S> m_vector;
  bool m_persistent_vector;

  StreamVector()
    : m_persistent_vector(false) {}

  StreamVector(const char* name, size_t max_size)
    : m_max_size(max_size) {

    if constexpr (!std::is_same_v<K::HostSpace, memory_space>)
      m_values_d =
        decltype(m_values_d)(
          K::ViewAllocateWithoutInitializing(name),
          m_max_size);
  }

  size_t
  max_size() const {
    return m_max_size;
  }

  size_t
  size() const {
    return m_values.is_allocated() ? m_values.extent(0) : 0;
  }

  void
  resize(size_t sz) {
    assert(sz < m_max_size);
    if constexpr (std::is_same_v<K::HostSpace, memory_space>) {
      m_vector.resize(sz);
      m_values_d =
        decltype(m_values_d)(
          reinterpret_cast<T*>(m_vector.data()),
          m_vector.size());
      m_values = m_values_d;
    } else {
      if (m_persistent_vector)
        m_vector.resize(sz);
      m_values = decltype(m_values)(m_values_d.data(), sz);
    }
  }

  void
  set(execution_space espace, bool persistent, std::vector<S>&& vector) {
    m_vector = std::move(vector);
    if constexpr (!std::is_same_v<K::HostSpace, memory_space>)
      assert(m_vector.size() <= m_values_d.extent(0));
    m_persistent_vector = persistent;
    if (m_vector.size() > 0) {
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        K::View<T*, K::HostSpace>
          values_h(reinterpret_cast<T*>(m_vector.data()), m_vector.size());
        m_values = decltype(m_values)(m_values_d.data(), m_vector.size());
        K::deep_copy(espace, m_values, values_h);
      } else {
        m_values_d =
          decltype(m_values_d)(
            reinterpret_cast<T*>(m_vector.data()),
            m_vector.size());
        m_values = m_values_d;
      }
    } else {
      m_values_d = decltype(m_values_d)();
      m_values = decltype(m_values)();
    }
  }

  void
  copy_from(const StreamVector& other, execution_space espace) {
    if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
      m_persistent_vector = other.m_persistent_vector;
      if (m_persistent_vector)
        m_vector = other.m_vector;
      m_values_d =
        decltype(m_values_d)(
          K::ViewAllocateWithoutInitializing(other.m_values_d.label()),
          other.m_values_d.extent(0));
      m_values =
        decltype(m_values)(m_values_d.data(), other.m_values.extent(0));
      K::deep_copy(espace, m_values, other.m_values);
    } else {
      m_persistent_vector = other.m_persistent_vector;
      m_vector = other.m_vector;
      m_values_d =
        decltype(m_values_d)(
          reinterpret_cast<T*>(m_vector.data()),
          m_vector.size());
      m_values = m_values_d;
    }
  }
};

/** container for data and views associated with one stream of an
 * ExecutionContext */
template <Device D>
struct /*HPG_EXPORT*/ StreamContext final {
  using kokkos_device = typename impl::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;
  using memory_space = typename execution_space::memory_space;
  using weight_traits =
    std::conditional_t<
      std::is_same_v<memory_space, K::HostSpace>,
      K::MemoryUnmanaged,
      K::MemoryManaged>;

  execution_space m_space;
  mutable State::maybe_vis_t m_vis_promise;
  size_t m_max_num_channels;

  template <unsigned N>
  using GVisVector =
    StreamVector<D, impl::core::poln_array_type<visibility_fp, N>>;
  std::variant<GVisVector<1>, GVisVector<2>, GVisVector<3>, GVisVector<4>>
    m_gvis;

  template <unsigned N>
  using VisVector = StreamVector<D, impl::visdata_t<N>, ::hpg::VisData<N>>;
  std::variant<VisVector<1>, VisVector<2>, VisVector<3>, VisVector<4>>
    m_visibilities;

  StreamVector<D, vis_weight_fp> m_weight_values;
  StreamVector<D, unsigned> m_weight_col_index;
  StreamVector<D, size_t> m_weight_row_index;

  std::any m_user_data;

  StreamContext() = delete;

  StreamContext(
    execution_space sp,
    unsigned npol,
    size_t max_num_vis,
    size_t max_num_channels)
    : m_space(sp)
    , m_max_num_channels(max_num_channels)
    , m_weight_values("weight_values", m_max_num_channels)
    , m_weight_col_index("weight_col_index", m_max_num_channels)
    , m_weight_row_index("weight_row_index", max_num_vis + 1) {

    assert(0 < npol && npol <= 4);
    const char* vbname = "visibility_buffer";
    const char* gvbname = "gvis_buffer";
    switch (npol) {
    case 1:
      m_visibilities = VisVector<1>(vbname, max_num_vis);
      m_gvis = GVisVector<1>(gvbname, max_num_vis);
      break;
    case 2:
      m_visibilities = VisVector<2>(vbname, max_num_vis);
      m_gvis = GVisVector<2>(gvbname, max_num_vis);
      break;
    case 3:
      m_visibilities = VisVector<3>(vbname, max_num_vis);
      m_gvis = GVisVector<3>(gvbname, max_num_vis);
      break;
    case 4:
      m_visibilities = VisVector<4>(vbname, max_num_vis);
      m_gvis = GVisVector<4>(gvbname, max_num_vis);
      break;
    default:
      std::abort();
      break;
    }
  }

  StreamContext(const StreamContext&) = delete;

  StreamContext(StreamContext&& other) noexcept = default;

  virtual ~StreamContext() {}

  StreamContext&
  operator=(StreamContext&& rhs) noexcept {
    fence();
    std::swap(*this, rhs);
    return *this;
  }

  StreamContext&
  operator=(const StreamContext&) = delete;

  size_t
  max_num_vis() const {
    return
      std::visit(
        overloaded {
          [](auto& v) { return v.max_size(); }
        },
        m_visibilities);
  }

  size_t
  num_vis() const {
    return
      std::visit(
        overloaded {
          [](auto& v) { return v.size(); }
            },
        m_visibilities);
  }

  size_t
  max_num_channels() const {
    return m_max_num_channels;
  }

  unsigned
  num_polarizations() const {
    return unsigned(m_visibilities.index()) + 1;
  }

  void
  copy_from(const StreamContext& other) {
    // caller must ensure that it's safe to overwrite values in this instance,
    // and to read values from 'other'
    assert(max_num_vis() == other.max_num_vis());
    assert(max_num_channels() == other.max_num_channels());

    m_max_num_channels = other.max_num_channels();

    std::visit(
      overloaded {
        [this, &other](auto& v) {
          using v_t = std::remove_reference_t<decltype(v)>;
          v.copy_from(std::get<v_t>(other.m_visibilities), m_space);
        }
      },
      m_visibilities);
    std::visit(
      overloaded {
        [this, &other](auto& v) {
          using v_t = std::remove_reference_t<decltype(v)>;
          v.copy_from(std::get<v_t>(other.m_gvis), m_space);
        }
      },
      m_gvis);
    m_weight_values.copy_from(other.m_weight_values, m_space);
    m_weight_col_index.copy_from(other.m_weight_col_index, m_space);
    m_weight_row_index.copy_from(other.m_weight_row_index, m_space);
  }

  template <unsigned N>
  constexpr VisVector<N>
  visibilities() {
    return std::get<VisVector<N>>(m_visibilities);
  }

  template <unsigned N>
  constexpr impl::visdata_view<N, memory_space>
  visdata() {
    return visibilities<N>().m_values;
  }

  template <unsigned N>
  constexpr GVisVector<N>
  gvis() {
    return std::get<GVisVector<N>>(m_gvis);
  }

  template <unsigned N>
  constexpr impl::gvis_view<N, memory_space>
  gvis_view() {
    return gvis<N>().m_values;
  }

  template <unsigned N>
  size_t
  copy_visibilities_to_device(std::vector<::hpg::VisData<N>>&& in_vis) {

    auto& v = std::get<VisVector<N>>(m_visibilities);
    v.set(m_space, true, std::move(in_vis));
    auto& gv = std::get<GVisVector<N>>(m_gvis);
    gv.resize(v.size());
    return v.size();
  }

  std::tuple<size_t, size_t>
  copy_weights_to_device(
    std::vector<vis_weight_fp>&& in_weight_values,
    std::vector<unsigned>&& in_weight_col_index,
    std::vector<size_t>&& in_weight_row_index) {

    m_weight_values.set(m_space, false, std::move(in_weight_values));
    m_weight_col_index.set(m_space, false, std::move(in_weight_col_index));
    m_weight_row_index.set(m_space, false, std::move(in_weight_row_index));

    return
      {m_weight_values.m_vector.size(), m_weight_row_index.m_vector.size()};
  }

  State::maybe_vis_t
  copy_visibilities_to_host(bool return_visibilities) {

    State::maybe_vis_t result;
    if (return_visibilities) {
      m_vis_promise =
        std::make_shared<std::shared_ptr<std::optional<VisDataVector>>>(
          std::make_shared<std::optional<VisDataVector>>(std::nullopt));
      result = m_vis_promise;
      if constexpr (!std::is_same_v<K::HostSpace, memory_space>) {
        std::visit(
          overloaded {
            [this](auto& v) {
              using dev_value_t = typename decltype(v.m_values)::value_type;
              if (v.size() > 0) {
                auto hview =
                  K::View<dev_value_t*, K::HostSpace, K::MemoryUnmanaged>(
                    reinterpret_cast<dev_value_t*>(v.m_vector.data()),
                    v.m_vector.size());
                auto dview =
                  K::subview(
                    visdata<dev_value_t::npol>(),
                    std::pair((size_t)0, v.m_vector.size()));
                K::deep_copy(m_space, hview, dview);
              }
            }
          },
          m_visibilities);
      }
    }
    return result;
  }

  void
  fence() const noexcept {
    m_space.fence();
    if (m_vis_promise) {
      std::visit(
        overloaded {
          [this](auto& v) {
            auto vdv =
              std::make_shared<std::optional<VisDataVector>>(
                VisDataVector(std::move(v.m_vector)));
            std::atomic_store(&*m_vis_promise, vdv);
          }
        },
        m_visibilities);
      m_vis_promise.reset();
    }
  }
};

template <Device D>
class ExecutionContextGroup;

template <Device D>
class /*HPG_EXPORT*/ ExecutionContext {
public:

  using pool_type = typename CFPoolRepo<D>::pool_type;
  using kokkos_device = typename impl::DeviceT<D>::kokkos_device;
  using execution_space = typename kokkos_device::execution_space;

protected:

  friend class ExecutionContextGroup<D>;

  struct StreamState {
    StreamContext<D> m_context;
    pool_type m_cf_pool;
    StreamPhase m_phase;
  };

  // CFPoolRepo instance should be owned by an instance of ExecutionContextGroup
  // (or similar)
  CFPoolRepo<D>* m_cf_pool_repo;

  std::deque<StreamState> m_streams;

public:
  using pool_id_type = typename CFPoolRepo<D>::id_type;

  ExecutionContext(
    unsigned npol,
    size_t max_num_vis,
    size_t max_num_channels,
    size_t num_streams,
    CFPoolRepo<D>* cf_pool_repo)
    : m_cf_pool_repo(cf_pool_repo) {

    assert(cf_pool_repo->size() >= num_streams);
    std::vector<int> weights(num_streams);
    std::fill_n(weights.begin(), weights.size(), 1);
    for (auto& s:
           K::Experimental::partition_space(execution_space(), weights))
      m_streams.push_back(
        StreamState{
          StreamContext<D>(
            s,
            npol,
            max_num_vis,
            max_num_channels),
          nullptr,
          StreamPhase::COPY});
  }

  ExecutionContext(ExecutionContext&& other)
    : m_cf_pool_repo(std::move(other).m_cf_pool_repo)
    , m_streams(std::move(other).m_streams) {}

  ExecutionContext&
  operator=(ExecutionContext&& rhs) {
    fence();
    std::swap(*this, rhs);
    return *this;
  }

  unsigned
  num_polarizations() const {
    if (m_streams.size() > 0)
      return m_streams.front().m_context.num_polarizations();
    return 0;
  }

  size_t
  num_streams() const {
    return m_streams.size();
  }

  template <typename F>
  void
  for_all_stream_contexts(F&& f) {
    for (auto& s : m_streams)
      f(s.m_context);
  }

  template <typename F, typename T>
  T
  fold_left_over_stream_contexts(const T& t, F&& f) {
    T result = t;
    for (auto& s : m_streams)
      result = f(result, s);
    return result;
  }

  template <typename F>
  void
  for_all_cf_pools(F&& f) {
    m_cf_pool_repo->for_all_cf_pools(
      [this, &f](auto& pool) {
        if (fold_left_over_stream_contexts(
              false,
              [&pool](bool match, auto& sc) {
                return match || sc.m_cf_pool == pool;
              }))
          f(pool);
      });
  }

  pool_type&
  current_cf_pool() {
    return m_streams.front().m_cf_pool;
  }

  const pool_type&
  current_cf_pool() const {
    return m_streams.front().m_cf_pool;
  }

  typename CFPoolRepo<D>::id_type
  current_cf_pool_id() const {
    return m_cf_pool_repo->get_id(current_cf_pool()).value();
  }

  pool_type
  find_pool(typename CFPoolRepo<D>::id_type id) const {
    return m_cf_pool_repo->find_pool(id);
  }

  StreamContext<D>&
  current_stream_context() {
    return m_streams.front().m_context;
  }

  const StreamContext<D>&
  current_stream_context() const {
    return m_streams.front().m_context;
  }

  execution_space&
  current_exec_space() {
    return current_stream_context().m_space;
  }

  void
  switch_to_copy(
    bool change_cf,
    const std::optional<pool_id_type>& id = std::nullopt) {

    if (m_streams.front().m_phase == StreamPhase::COMPUTE) {
      next_stream();
      if (change_cf)
        next_cf_pool(id);
    } else if (change_cf && id) {
      next_cf_pool(id);
    }
    // don't need to do anything if the head stream is in COPY phase and the
    // caller asked for a new CFPool, as in that case it's OK to simply allow
    // the caller to overwrite the current CFPool instead of using another
    // instance
    m_streams.front().m_phase = StreamPhase::COPY;
}

  void
  switch_to_compute() {
    m_streams.front().m_phase = StreamPhase::COMPUTE;
  }

  void
  next_stream() {
    // rotate the head stream to the tail
    m_streams.push_back(std::move(m_streams.front()));
    m_streams.pop_front();
    // set new head stream CFPool to same instance as previous head stream
    m_streams.front().m_cf_pool = m_streams.back().m_cf_pool;
  }

  void
  next_cf_pool(const std::optional<pool_id_type>& id = std::nullopt) {
    pool_type new_pool;
    // try to acquire CFPool with given id
    if (id)
      new_pool = m_cf_pool_repo->find_pool(id.value());
    // try to acquire an unused CFPool (and set its id)
    if (!new_pool)
      new_pool = m_cf_pool_repo->get_unused_pool(id);
    if (!new_pool) {
      // no CFPool is available -- fence the current stream and all other
      // streams using the same CFPool instance, and remove that CFPool from all
      // streams, then it becomes unused and the this stream can acquire it
      {
        pool_type current_pool = m_streams.front().m_cf_pool;
        for (auto& s : m_streams)
          if (s.m_cf_pool && s.m_cf_pool == current_pool) {
            s.m_context.fence();
            s.m_cf_pool.reset();
          }
      }
      new_pool = m_cf_pool_repo->get_unused_pool(id);
      assert(new_pool);
    }
    m_streams.front().m_cf_pool = new_pool;
  }

  void
  fence() const {
    for (auto& s : m_streams)
      s.m_context.fence();
  }
};

template <Device D>
class /*HPG_EXPORT*/ ExecutionContextGroup {

  CFPoolRepo<D> m_cf_pool_repo;

  std::vector<ExecutionContext<D>> m_execution_contexts;

  using execution_space = typename ExecutionContext<D>::execution_space;

public:

  using size_type = typename decltype(m_execution_contexts)::size_type;

  ExecutionContextGroup() {}

  ExecutionContextGroup(
    unsigned npol,
    size_t max_num_vis,
    size_t max_num_channels,
    const std::vector<size_t>& num_streams,
    size_t pool_size)
    : m_cf_pool_repo(pool_size) {

    for (auto& n: num_streams)
      m_execution_contexts
        .emplace_back(npol, max_num_vis, max_num_channels, n, &m_cf_pool_repo);
    for (auto& ec : m_execution_contexts)
      ec.next_cf_pool();
  }

  ExecutionContextGroup(const ExecutionContextGroup& other)
    : m_cf_pool_repo(other.m_cf_pool_repo.size()) {

    other.fence();
    execution_space espace;
    auto npol = other.num_polarizations();
    auto cf_map = m_cf_pool_repo.copy_from(espace, other.m_cf_pool_repo);
    auto max_num_vis = other[0].current_stream_context().max_num_vis();
    auto max_num_channels =
      other[0].current_stream_context().max_num_channels();
    for (auto& ec: other.m_execution_contexts) {
      m_execution_contexts.emplace_back(
        npol,
        max_num_vis,
        max_num_channels,
        ec.num_streams(),
        &m_cf_pool_repo);
      auto& new_ct = m_execution_contexts.back();
      for (size_t i = 0; i < ec.num_streams(); ++i) {
        auto& src = ec.m_streams[i];
        auto& dst = new_ct.m_streams[i];
        dst.m_context.copy_from(src.m_context);
        if (src.m_cf_pool)
          dst.m_cf_pool = cf_map.at(src.m_cf_pool.get()).lock();
      }
    }
    espace.fence();
  }

  ExecutionContextGroup(ExecutionContextGroup&& other)
    : m_cf_pool_repo(std::move(other).m_cf_pool_repo)
    , m_execution_contexts(std::move(other).m_execution_contexts) {

    for (auto& ec : m_execution_contexts)
      ec.m_cf_pool_repo = &m_cf_pool_repo;
  }

  ExecutionContextGroup&
  operator=(ExecutionContextGroup&& rhs) {

    m_cf_pool_repo = std::move(rhs).m_cf_pool_repo;
    m_execution_contexts = std::move(rhs).m_execution_contexts;
    for (auto& ec : m_execution_contexts)
      ec.m_cf_pool_repo = &m_cf_pool_repo;
    return *this;
  }

  ExecutionContextGroup&
  operator=(const ExecutionContextGroup& rhs) {
    ExecutionContextGroup tmp(rhs);
    fence();
    std::swap(*this, tmp);
    return *this;
  }

  size_type
  size() const {
    return m_execution_contexts.size();
  }

  ExecutionContext<D>&
  operator[](size_type i) {
    return m_execution_contexts[i];
  }

  const ExecutionContext<D>&
  operator[](size_type i) const {
    return m_execution_contexts[i];
  }

  unsigned
  num_polarizations() const {
    if (size() > 0)
      return m_execution_contexts[0].num_polarizations();
    return 0;
  }

  void
  switch_to_copy() {
    for (auto& ec : m_execution_contexts)
      ec.switch_to_copy(false);
  }

  void
  switch_to_compute() {
    for (auto& ec : m_execution_contexts)
      ec.switch_to_compute();
  }

  template <typename F>
  void
  for_all_stream_contexts(F&& f) {
    for (auto& ec : m_execution_contexts)
      ec.for_all_stream_contexts(f);
  }

  template <typename F, typename T>
  T
  fold_left_over_stream_contexts(const T& t, F&& f) {
    T result = t;
    for (auto& ec : m_execution_contexts)
      result = ec.fold_left_over_stream_contexts(result, f);
    return result;
  }

  template <typename F>
  void
  for_all_cf_pools(F&& f) {
    m_cf_pool_repo.for_all_cf_pools(std::forward<F>(f));
  }

  unsigned
  total_streams() const {
    unsigned result = 0;
    for (auto& ec : m_execution_contexts)
      result += ec.num_streams();
    return result;
  }

  void
  fence() const {
    for (auto& ec : m_execution_contexts)
      ec.fence();
  }
};

/** Kokkos state implementation for a device type */
template <Device D>
struct /*HPG_EXPORT*/ StateT
  : virtual public State {
public:

  Device m_device; /**< device type */
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

protected:

  static std::vector<size_t>
  repeated_value(unsigned n, size_t v) {
    std::vector<size_t> result(n);
    std::fill(result.begin(), result.end(), v);
    return result;
  }

  static unsigned
  limit_tasks(unsigned n) {
    return std::min(n, device_traits::active_task_limit);
  }

  mutable std::mutex m_mtx;
  // access to the following members in const methods must be protected by m_mtx
  // (intentionally do not provide any thread safety guarantee outside of const
  // methods!)
  mutable ExecutionContextGroup<D> m_exec_contexts;

public:

  StateT(
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size_global,
    const std::array<unsigned, 4>& grid_offset_local,
    const std::array<unsigned, 4>& grid_size_local,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions,
    ExecutionContextGroup<D>&& exec_contexts)
    : m_device(D)
    , m_visibility_batch_size(visibility_batch_size)
    , m_max_avg_channels_per_vis(max_avg_channels_per_vis)
    , m_grid_scale({grid_scale[0], grid_scale[1]})
    , m_num_polarizations(mueller_indexes.m_npol)
    , m_implementation_versions(implementation_versions)
    , m_exec_contexts(std::move(exec_contexts)) {

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

    if (init_cf_shape)
      init_cfs(*init_cf_shape);
    m_mueller_indexes =
      init_mueller("mueller_indexes", mueller_indexes);
    m_conjugate_mueller_indexes =
      init_mueller("conjugate_mueller_indexes", conjugate_mueller_indexes);
    new_grid(true, true);
  }

  StateT(
    unsigned num_contexts,
    unsigned max_active_tasks_per_context,
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
  : StateT(
    visibility_batch_size,
    max_avg_channels_per_vis,
    init_cf_shape,
    grid_size_global,
    grid_offset_local,
    grid_size_local,
    grid_scale,
    mueller_indexes,
    conjugate_mueller_indexes,
    implementation_versions,
    ExecutionContextGroup<D>(
      mueller_indexes.m_npol,
      visibility_batch_size,
      max_avg_channels_per_vis * visibility_batch_size,
      repeated_value(
        num_contexts,
        limit_tasks(max_active_tasks_per_context)),
      num_contexts * limit_tasks(max_active_tasks_per_context))) {}

  StateT(const StateT& st)
    : m_device(D)
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
    m_exec_contexts = st.m_exec_contexts;
    copy_model(st);
    m_mueller_indexes = st.m_mueller_indexes;
    m_conjugate_mueller_indexes = st.m_conjugate_mueller_indexes;
    new_grid(&st, true);
  }

  StateT(StateT&& st) noexcept
    : m_device(D)
    , m_visibility_batch_size(std::move(st).m_visibility_batch_size)
    , m_max_avg_channels_per_vis(std::move(st).m_max_avg_channels_per_vis)
    , m_grid_size_global(std::move(st).m_grid_size_global)
    , m_grid_offset_local(std::move(st).m_grid_offset_local)
    , m_grid_size_local(std::move(st).m_grid_size_local)
    , m_grid_scale(std::move(st).m_grid_scale)
    , m_num_polarizations(std::move(st).m_num_polarizations)
    , m_implementation_versions(std::move(st).m_implementation_versions)
    , m_grid(std::move(st).m_grid)
    , m_grid_weights(std::move(st).m_grid_weights)
    , m_model(std::move(st).m_model)
    , m_mueller_indexes(std::move(st).m_mueller_indexes)
    , m_conjugate_mueller_indexes(std::move(st).m_conjugate_mueller_indexes)
    , m_exec_contexts(std::move(st).m_exec_contexts) {}

  virtual ~StateT() {
    fence();
    m_grid = decltype(m_grid)();
    m_grid_weights = decltype(m_grid_weights)();
    m_model = decltype(m_model)();
    m_mueller_indexes = decltype(m_mueller_indexes)();
    m_conjugate_mueller_indexes = decltype(m_conjugate_mueller_indexes)();
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
  num_exec_contexts() const noexcept override {
    return m_exec_contexts.size();
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
  num_active_tasks() const noexcept override {
    return m_exec_contexts.total_streams();
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
  convolution_function_region_size(const CFArrayShape& shape)
    const noexcept override {
    std::scoped_lock lock(m_mtx);
    return m_exec_contexts[0].current_cf_pool()->pool_size(shape);
  }

  virtual rval_t<size_t>
  current_convolution_function_region_size(unsigned context)
    const noexcept override {

    if (context >= m_exec_contexts.size())
      return rval<size_t>(std::make_unique<InvalidGriddingContextError>());

    std::scoped_lock lock(m_mtx);
    return
      rval<size_t>(
        m_exec_contexts[context].current_cf_pool()->pool.extent(0));
  }

  virtual std::optional<std::unique_ptr<Error>>
  allocate_convolution_function_region(const CFArrayShape& shape) override {

    m_exec_contexts.fence();
    m_exec_contexts.for_all_cf_pools(
      [shape](auto& cf) {
        cf->prepare_pool(shape, true);
      });
    return std::nullopt;
  }

  virtual std::optional<std::unique_ptr<Error>>
  set_convolution_function(
    unsigned context,
    Device host_device,
    CFArray&& cf_array) override {

    if (context >= m_exec_contexts.size())
      return
        std::make_optional(std::make_unique<InvalidGriddingContextError>());

    m_exec_contexts[context].switch_to_copy(true);
    auto& espace = m_exec_contexts[context].current_exec_space();
    auto& cf = m_exec_contexts[context].current_cf_pool();
    try {
      cf->add_device_cfs(
        espace,
        std::move(dynamic_cast<typename impl::DeviceCFArray<D>&&>(cf_array)));
    } catch (const std::bad_cast&) {
      cf->add_host_cfs(host_device, espace, std::move(cf_array));
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
    m_exec_contexts.switch_to_copy();
    if (!m_model.is_allocated())
      m_model =
        decltype(m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          grid_layout::dimensions(m_grid_size_local));

    try {
      impl::GridValueViewArray<D> gvv =
        std::move(dynamic_cast<impl::GridValueViewArray<D>&&>(gv));
      K::deep_copy(m_exec_contexts[0].current_exec_space(), m_model, gvv.grid);
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
      K::deep_copy(m_exec_contexts[0].current_exec_space(), m_model, model_h);
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

    m_exec_contexts[context].switch_to_copy(false);
    auto& sc_copy = m_exec_contexts[context].current_stream_context();
    int len = sc_copy.copy_visibilities_to_device(std::move(visibilities));
    sc_copy.copy_weights_to_device(
      std::move(wgt_values),
      std::move(wgt_col_index),
      std::move(wgt_row_index));

    m_exec_contexts[context].switch_to_compute();
    auto& sc_grid = m_exec_contexts[context].current_stream_context();
    auto& cf = m_exec_contexts[context].current_cf_pool();
    auto gvis = sc_grid.template gvis_view<N>();
    using gvis0 =
      typename std::remove_reference_t<decltype(gvis)>::value_type;
    K::deep_copy(sc_grid.m_space, gvis, gvis0());

    auto gridder =
      impl::core::VisibilityGridder(
        sc_grid.m_space,
        cf->cf_d,
        cf->cf_radii,
        cf->max_cf_extent_y,
        m_mueller_indexes,
        m_conjugate_mueller_indexes,
        len,
        0,
        1,
        sc_grid.template visdata<N>(),
        sc_grid.m_weight_values.m_values,
        sc_grid.m_weight_col_index.m_values,
        sc_grid.m_weight_row_index.m_values,
        gvis,
        m_grid_scale,
        m_grid,
        m_grid_weights,
        m_model,
        m_grid_offset_local,
        m_grid_size_global);

    if (do_degrid) {
      gridder.degrid_all();
      if (do_grid)
        gridder.vis_copy_residual_and_rescale();
      else
        gridder.vis_copy_predicted();
    } else {
      gridder.vis_rescale();
    }

    if (do_grid) {
      if (update_grid_weights)
        gridder.grid_all();
      else
        gridder.grid_all_no_weights();
    }
    return sc_grid.copy_visibilities_to_host(return_visibilities);
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
        default_grid_visibilities<N>(
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

    if (context >= m_exec_contexts.size())
      return std::make_unique<InvalidGriddingContextError>();

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
    auto g_h = K::create_mirror_view(m_grid);
    K::deep_copy(g_h, impl::gv_t(0));
    m_exec_contexts.switch_to_copy();
    K::deep_copy(m_exec_contexts[0].current_exec_space(), m_grid, g_h);
  };

  virtual void
  fill_grid_weights(const grid_value_fp& val) override {
    auto w_h = K::create_mirror_view(m_grid_weights);
    K::deep_copy(w_h, grid_value_fp(0));
    m_exec_contexts.switch_to_copy();
    K::deep_copy(m_exec_contexts[0].current_exec_space(), m_grid_weights, w_h);
  };

  virtual void
  reset_model() override {
    fence();
    m_model = decltype(m_model)();
  }

  virtual void
  normalize_by_weights(grid_value_fp wfactor) override {
    m_exec_contexts.switch_to_compute();
    impl::core::GridNormalizer(
      m_exec_contexts[0].current_exec_space(),
      m_grid,
      m_grid_weights,
      wfactor)
      .normalize();
  }

  virtual std::optional<std::unique_ptr<Error>>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    m_exec_contexts.switch_to_compute();
    std::optional<std::unique_ptr<Error>> err;
    if (in_place) {
      switch (fft_version()) {
      case 0:
        err =
          impl::FFT<execution_space>::in_place_kernel(
            m_exec_contexts[0].current_exec_space(),
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
            m_exec_contexts[0].current_exec_space(),
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
      m_exec_contexts[0].current_exec_space(),
      m_grid,
      norm)
      .normalize();
    return err;
  }

  virtual std::optional<std::unique_ptr<Error>>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {

    m_exec_contexts.switch_to_compute();
    std::optional<std::unique_ptr<Error>> err;
    if (m_model.is_allocated()){
      if (in_place) {
        switch (fft_version()) {
        case 0:
          err =
            impl::FFT<execution_space>::in_place_kernel(
              m_exec_contexts[0].current_exec_space(),
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
              m_exec_contexts[0].current_exec_space(),
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
        m_exec_contexts[0].current_exec_space(),
        m_model,
        norm)
        .normalize();
    }
    return err;
  }

  virtual void
  shift_grid(ShiftDirection direction) override {
    m_exec_contexts.switch_to_compute();
    impl::core::GridShifter(
      m_exec_contexts[0].current_exec_space(),
      direction,
      m_grid)
      .shift();
  }

  virtual void
  shift_model(ShiftDirection direction) override {
    m_exec_contexts.switch_to_compute();
    impl::core::GridShifter(
      m_exec_contexts[0].current_exec_space(),
      direction,
      m_model)
      .shift();
  }

protected:

  virtual void
  fence_unlocked() const noexcept {
    m_exec_contexts.fence();
  }

  std::unique_ptr<GridWeightArray>
  grid_weights_unlocked() const {
    fence_unlocked();
    auto wgts_h = K::create_mirror(m_grid_weights);
    m_exec_contexts.switch_to_copy();
    K::deep_copy(
      m_exec_contexts[0].current_exec_space(),
      wgts_h,
      m_grid_weights);
    m_exec_contexts[0].current_stream_context().fence();
    return std::make_unique<impl::GridWeightViewArray<D>>(wgts_h);
  }

  std::unique_ptr<GridValueArray>
  grid_values_unlocked() const noexcept {
    fence_unlocked();
    auto grid_h = K::create_mirror(m_grid);
    m_exec_contexts.switch_to_copy();
    K::deep_copy(m_exec_contexts[0].current_exec_space(), grid_h, m_grid);
    m_exec_contexts[0].current_stream_context().fence();
    return std::make_unique<impl::GridValueViewArray<D>>(grid_h);
  }

  std::unique_ptr<GridValueArray>
  model_values_unlocked() const {
    fence_unlocked();
    if (m_model.is_allocated()) {
      auto model_h = K::create_mirror(m_model);
      m_exec_contexts.switch_to_copy();
      K::deep_copy(m_exec_contexts[0].current_exec_space(), model_h, m_model);
      m_exec_contexts[0].current_stream_context().fence();
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
    std::swap(m_visibility_batch_size, other.m_visibility_batch_size);
    std::swap(
      m_max_avg_channels_per_vis,
      other.m_max_avg_channels_per_vis);
    std::swap(m_grid_size_global, other.m_grid_size_global);
    std::swap(m_grid_size_local, other.m_grid_size_local);
    std::swap(m_grid_offset_local, other.m_grid_offset_local);
    std::swap(m_grid_scale, other.m_grid_scale);
    std::swap(m_implementation_versions, other.m_implementation_versions);
    std::swap(m_exec_contexts[0], other.m_exec_contexts[0]);

    std::swap(m_grid, other.m_grid);
    std::swap(m_grid_weights, other.m_grid_weights);
    std::swap(m_model, other.m_model);
    std::swap(m_mueller_indexes, other.m_mueller_indexes);
    std::swap(m_conjugate_mueller_indexes, other.m_conjugate_mueller_indexes);
  }

  void
  init_cfs(const CFArrayShape& init_cf_shape) {
    m_exec_contexts.switch_to_copy();
    m_exec_contexts.for_all_cf_pools(
      [&init_cf_shape](auto& p) {
        p->prepare_pool(init_cf_shape, true);
      });
  }

  void
  copy_model(const StateT& st) {
    m_exec_contexts.switch_to_copy();
    if (st.m_model.is_allocated()) {
      m_model =
        decltype(m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          grid_layout::dimensions(m_grid_size_local));
      K::deep_copy(
        m_exec_contexts[0].current_exec_space(),
        m_model,
        st.m_model);
    }
  }

  /** copy Mueller indexes to device */
  template <size_t N>
  impl::mindex_view<memory_space>
  copy_mueller_indexes(
    const std::string& name,
    const std::vector<iarray<N>>& mindexes) {

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
    m_exec_contexts.switch_to_copy();
    K::deep_copy(
      m_exec_contexts[0].current_exec_space(),
      result,
      mueller_indexes_h);
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

  void
  new_grid(std::variant<const StateT*, bool> source, bool also_weights) {

    const bool create_without_init =
      std::holds_alternative<const StateT*>(source) || !std::get<bool>(source);
    if (!create_without_init)
      m_exec_contexts.switch_to_copy();

    if (create_without_init)
      m_grid =
        decltype(m_grid)(
          K::ViewAllocateWithoutInitializing("grid"),
          grid_layout::dimensions(m_grid_size_local));
    else
      m_grid =
        decltype(m_grid)(
          K::view_alloc("grid", m_exec_contexts[0].current_exec_space()),
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
              m_exec_contexts[0].current_exec_space()),
            int(m_grid_size_local[int(impl::core::GridAxis::mrow)]),
            int(m_grid_size_local[int(impl::core::GridAxis::channel)]));
    }
    if (std::holds_alternative<const StateT*>(source)) {
      auto st = std::get<const StateT*>(source);
      K::deep_copy(m_exec_contexts[0].current_exec_space(), m_grid, st->m_grid);
      if (also_weights)
        K::deep_copy(
          m_exec_contexts[0].current_exec_space(),
          m_grid_weights,
          st->m_grid_weights);
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
  allocate_convolution_function_region(
    GS&& st,
    const CFArrayShape& shape) {

    ::hpg::GridderState result(std::forward<GS>(st));
    if (auto error =
        result.m_impl->allocate_convolution_function_region(shape);
        error)
      return std::move(error.value());
    else
      return std::move(result);
  }

  template <typename GS>
  static std::variant<std::unique_ptr<Error>, ::hpg::GridderState>
  set_convolution_function(
    GS&& st,
    unsigned context,
    Device host_device,
    CFArray&& cf) {

    if (host_devices().count(host_device) > 0) {
      ::hpg::GridderState result(std::forward<GS>(st));
      if (auto error =
          result.m_impl
            ->set_convolution_function(context, host_device, std::move(cf));
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
    unsigned context,
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
        context,
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
