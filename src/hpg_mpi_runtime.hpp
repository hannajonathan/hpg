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

#include "hpg_mpi.hpp"
#include "hpg_runtime.hpp"
// #include "hpg_export.h"


#include <Kokkos_Core.hpp>

namespace hpg::mpi::runtime {

namespace impl = ::hpg::runtime::impl;

namespace K = Kokkos;

template <typename T>
struct mpi_datatype {
  using scalar_type = T;
  using value_type = void;
  static value_type value() {};
};
template <>
struct mpi_datatype<char> {
  using scalar_type = char;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CHAR; };
};
template <>
struct mpi_datatype<signed short int> {
  using scalar_type = signed short int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_SHORT; }
};
template <>
struct mpi_datatype<signed int> {
  using scalar_type = signed int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_INT; }
};
template <>
struct mpi_datatype<signed long int> {
  using scalar_type = signed long int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_LONG; }
};
template <>
struct mpi_datatype<signed long long int> {
  using scalar_type = signed long long int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_LONG_LONG; }
};
template <>
struct mpi_datatype<signed char> {
  using scalar_type = signed char;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_SIGNED_CHAR; }
};
template <>
struct mpi_datatype<unsigned char> {
  using scalar_type = unsigned char;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_CHAR; }
};
template <>
struct mpi_datatype<unsigned short int> {
  using scalar_type = unsigned short int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_SHORT; }
};
template <>
struct mpi_datatype<unsigned int> {
  using scalar_type = unsigned int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED; }
};
template <>
struct mpi_datatype<unsigned long int> {
  using scalar_type = unsigned long int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_LONG; }
};
template <>
struct mpi_datatype<unsigned long long int> {
  using scalar_type = unsigned long long int;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_LONG_LONG; }
};
template <>
struct mpi_datatype<float> {
  using scalar_type = float;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_FLOAT; }
};
template <>
struct mpi_datatype<double> {
  using scalar_type = double;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_DOUBLE; }
};
template <>
struct mpi_datatype<long double> {
  using scalar_type = long double;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_LONG_DOUBLE; }
};
template <>
struct mpi_datatype<wchar_t> {
  using scalar_type = wchar_t;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_WCHAR; }
};
template <>
struct mpi_datatype<bool> {
  using scalar_type = bool;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_BOOL; }
};
template <>
struct mpi_datatype<std::complex<float>> {
  using scalar_type = std::complex<float>;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_COMPLEX; }
};
template <>
struct mpi_datatype<std::complex<double>> {
  using scalar_type = std::complex<double>;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_DOUBLE_COMPLEX; }
};
template <>
struct mpi_datatype<std::complex<long double>> {
  using scalar_type = std::complex<long double>;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_LONG_DOUBLE_COMPLEX; }
};
template <>
struct mpi_datatype<K::complex<float>> {
  using scalar_type = K::complex<float>;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_COMPLEX; }
};
template <>
struct mpi_datatype<K::complex<double>> {
  using scalar_type = K::complex<double>;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_DOUBLE_COMPLEX; }
};
template <>
struct mpi_datatype<K::complex<long double>> {
  using scalar_type = K::complex<long double>;
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_LONG_DOUBLE_COMPLEX; }
};
template <unsigned N>
struct mpi_datatype<::hpg::VisData<N>> {
  using scalar_type = ::hpg::VisData<N>;
  using value_type = MPI_Datatype;
  static value_type
  value() {
    static std::once_flag flag;
    static MPI_Datatype result;
    std::call_once(
      flag,
      [](MPI_Datatype* dt) {
        constexpr int count = 6;
        const std::array<int, count> blocklengths{
          scalar_type::npol, // m_visibilities
          1, // m_frequency
          1, // m_phase
          std::tuple_size<decltype(scalar_type::m_uvw)>::value,
          std::tuple_size<decltype(scalar_type::m_cf_index)>::value,
          std::tuple_size<decltype(scalar_type::m_cf_phase_gradient)>::value
        };
        const std::array<MPI_Aint, count> displacements{
          offsetof(scalar_type, m_visibilities),
          offsetof(scalar_type, m_frequency),
          offsetof(scalar_type, m_phase),
          offsetof(scalar_type, m_uvw),
          offsetof(scalar_type, m_cf_index),
          offsetof(scalar_type, m_cf_phase_gradient)
        };
        const std::array<MPI_Datatype, count> types{
          mpi_datatype<
            typename decltype(scalar_type::m_visibilities)::value_type>
            ::value(),
          mpi_datatype<decltype(scalar_type::m_frequency)>::value(),
          mpi_datatype<decltype(scalar_type::m_phase)>::value(),
          mpi_datatype<
            typename decltype(scalar_type::m_uvw)::value_type>::value(),
          mpi_datatype<
            typename decltype(scalar_type::m_cf_index)::value_type>::value(),
          mpi_datatype<
            typename decltype(scalar_type::m_cf_phase_gradient)::value_type>
            ::value()
        };
        MPI_Type_create_struct(
          count,
          blocklengths.data(),
          displacements.data(),
          types.data(),
          dt);
        MPI_Type_commit(dt);
      },
      &result);
    return result;
  }
};
template <typename T, int N>
struct mpi_datatype<impl::core::poln_array_type<T, N>> {
  using scalar_type = impl::core::poln_array_type<T, N>;
  using value_type = MPI_Datatype;
  static value_type
  value() {
    static std::once_flag flag;
    static MPI_Datatype result;
    std::call_once(
      flag,
      [](MPI_Datatype* dt) {
        MPI_Type_contiguous(
          N,
          mpi_datatype<std::complex<T>>::value(),
          dt);
        MPI_Type_commit(dt);
      },
      &result);
    return result;
  }
};
template <
  unsigned N,
  typename V,
  typename F,
  typename P,
  typename U,
  typename G>
struct mpi_datatype<impl::core::VisData<N, V, F, P, U, G>> {
  using scalar_type = impl::core::VisData<N, V, F, P, U, G>;
  using value_type = MPI_Datatype;
  static value_type
  value() {
    static std::once_flag flag;
    static MPI_Datatype result;
    std::call_once(
      flag,
      [](MPI_Datatype* dt) {
        constexpr int count = 6;
        const std::array<int, count> blocklengths{
          scalar_type::npol, // m_values
          1, // m_freq
          1, // m_d_phase
          decltype(scalar_type::m_uvw)::size(),
          decltype(scalar_type::m_cf_index)::size(),
          decltype(scalar_type::m_cf_phase_gradient)::size()
        };
        const std::array<MPI_Aint, count> displacements{
          offsetof(scalar_type, m_values),
          offsetof(scalar_type, m_freq),
          offsetof(scalar_type, m_d_phase),
          offsetof(scalar_type, m_uvw),
          offsetof(scalar_type, m_cf_index),
          offsetof(scalar_type, m_cf_phase_gradient)
        };
        const std::array<MPI_Datatype, count> types{
          mpi_datatype<typename decltype(scalar_type::m_values)::value_type>
            ::value(),
          mpi_datatype<decltype(scalar_type::m_freq)>::value(),
          mpi_datatype<decltype(scalar_type::m_d_phase)>::value(),
          mpi_datatype<
            typename decltype(scalar_type::m_uvw)::value_type>::value(),
          mpi_datatype<
            typename decltype(scalar_type::m_cf_index)::value_type>::value(),
          mpi_datatype<
            typename decltype(scalar_type::m_cf_phase_gradient)::value_type>
            ::value()
        };
        MPI_Type_create_struct(
          count,
          blocklengths.data(),
          displacements.data(),
          types.data(),
          dt);
        MPI_Type_commit(dt);
      },
      &result);
    return result;
  }
};

struct ReplicatedGridBrick {
  // three axes are x, y, and channel; mrow partition not supported
  static constexpr unsigned rank = 3;
  enum Axis {x, y, channel};

  unsigned num_replicas;
  std::array<unsigned, rank> offset;
  std::array<unsigned, rank> size;
};

struct DevCFShape
  : public CFArrayShape {

  unsigned m_oversampling;

  std::vector<std::array<unsigned, rank - 1>> m_extents;

  DevCFShape(const std::vector<unsigned>& shape) {
    assert(shape.size() > 0);
    assert((shape.size() - 1) % (rank - 1) == 0);

    m_oversampling = shape[0];
    m_extents.reserve((shape.size() - 1) / (rank - 1));
    for (size_t grp = 0; grp < m_extents.size(); ++grp) {
      std::array<unsigned, rank - 1> ext;
      for (unsigned d = 0; d < rank - 1; ++d)
        ext[d] = shape[grp * (rank - 1) + d + 1];
      m_extents.push_back(ext);
    }
  }

  unsigned
  oversampling() const override {
    return m_oversampling;
  }

  unsigned
  num_groups() const override {
    return unsigned(m_extents.size());
  }

  std::array<unsigned, rank - 1>
  extents(unsigned grp) const override {
    return m_extents[grp];
  }
};

struct /*HPG_EXPORT*/ State
  : virtual public ::hpg::runtime::State {

protected:

  // communicator for subspace of visibility partition for this element of grid
  // partition
  MPI_Comm m_vis_comm;

  // communicator for subspace of grid plane partition for this element of
  // visibility partition
  MPI_Comm m_grid_comm;

  MPI_Comm m_replica_comm;

  MPI_Comm m_plane_comm;

  mutable bool m_reduced_grid;

  mutable bool m_reduced_weights;

  State()
    : m_vis_comm(MPI_COMM_NULL)
    , m_grid_comm(MPI_COMM_NULL)
    , m_replica_comm(MPI_COMM_NULL)
    , m_plane_comm(MPI_COMM_NULL)
    , m_reduced_grid(true)
    , m_reduced_weights(true) {}

public:

  State(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    MPI_Comm replica_comm,
    MPI_Comm plane_comm)
    : m_vis_comm(vis_comm)
    , m_grid_comm(grid_comm)
    , m_replica_comm(replica_comm)
    , m_plane_comm(plane_comm)
    , m_reduced_grid(true)
    , m_reduced_weights(true) {}

  virtual ~State() {
    for (auto& c :
           {&m_vis_comm, &m_grid_comm, &m_replica_comm, &m_plane_comm}) {
      if (*c != MPI_COMM_NULL && *c != MPI_COMM_SELF && *c != MPI_COMM_WORLD)
        MPI_Comm_free(c);
    }
  }

  bool
  non_trivial_visibility_partition() const noexcept {
    int size;
    MPI_Comm_size(m_vis_comm, &size);
    return size > 1;
  }

  bool
  is_visibility_partition_root() const noexcept {
    int rank;
    MPI_Comm_rank(m_vis_comm, &rank);
    return rank == 0;
  }

  bool
  non_trivial_grid_partition() const noexcept {
    int size;
    MPI_Comm_size(m_grid_comm, &size);
    return size > 1;
  }

  bool
  is_grid_partition_root() const noexcept {
    int rank;
    MPI_Comm_rank(m_grid_comm, &rank);
    return rank == 0;
  }

  bool
  non_trivial_replica_partition() const noexcept {
    int size;
    MPI_Comm_size(m_replica_comm, &size);
    return size > 1;
  }

  bool
  is_replica_partition_root() const noexcept {
    int rank;
    MPI_Comm_rank(m_replica_comm, &rank);
    return rank == 0;
  }

  bool
  non_trivial_plane_partition() const noexcept {
    return
      m_plane_comm != MPI_COMM_NULL
      && [c=m_plane_comm]{
        int sz;
        MPI_Comm_size(c, &sz);
        return sz > 1;
      }();
  }

  bool
  is_plane_partition_root() const noexcept {
    return m_plane_comm != MPI_COMM_NULL;
  }
};

template <Device D>
struct /*HPG_EXPORT*/ StateTBase
  : public ::hpg::runtime::StateT<D>
  , virtual public State {

  using typename ::hpg::runtime::StateT<D>::kokkos_device;
  using typename ::hpg::runtime::StateT<D>::execution_space;
  using typename ::hpg::runtime::StateT<D>::memory_space;
  using typename ::hpg::runtime::StateT<D>::device_traits;
  using typename ::hpg::runtime::StateT<D>::stream_type;
  using typename ::hpg::runtime::StateT<D>::grid_layout;

  using SD = typename ::hpg::runtime::StateT<D>;

  using StreamPhase = ::hpg::runtime::StreamPhase;

  // needed because of conversion from ReplicatedGridBrick array indexes to
  // GridAxis indexes in constructor
  static_assert(
    int(impl::core::GridAxis::x) == 0
    && int(impl::core::GridAxis::y) == 1
    && int(impl::core::GridAxis::mrow) == 2
    && int(impl::core::GridAxis::channel) == 3);

public:

  StateTBase(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    const ReplicatedGridBrick& grid_brick,
    MPI_Comm replica_comm,
    MPI_Comm plane_comm,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions,
    ::hpg::runtime::ExecutionContextGroup<D>&& exec_contexts)
    : State(vis_comm, grid_comm, replica_comm, plane_comm)
    , ::hpg::runtime::StateT<D>(
      visibility_batch_size,
      max_avg_channels_per_vis,
      init_cf_shape,
      grid_size,
      {grid_brick.offset[ReplicatedGridBrick::Axis::x],
       grid_brick.offset[ReplicatedGridBrick::Axis::y],
       0,
       grid_brick.offset[ReplicatedGridBrick::Axis::channel]},
      {grid_brick.size[ReplicatedGridBrick::Axis::x],
       grid_brick.size[ReplicatedGridBrick::Axis::y],
       grid_size[GridValueArray::Axis::mrow],
       grid_brick.size[ReplicatedGridBrick::Axis::channel]},
      grid_scale,
      mueller_indexes,
      conjugate_mueller_indexes,
      implementation_versions,
      std::move(exec_contexts)) {}

  StateTBase(const StateTBase& st)
    : State()
    , ::hpg::runtime::StateT<D>(st) {

    if (st.m_vis_comm != MPI_COMM_NULL) {
      if (st.m_vis_comm != MPI_COMM_SELF)
        MPI_Comm_dup(st.m_vis_comm, &m_vis_comm);
      else
        m_vis_comm = MPI_COMM_SELF;
    }
    if (st.m_grid_comm != MPI_COMM_NULL) {
      if (st.m_grid_comm != MPI_COMM_SELF)
        MPI_Comm_dup(st.m_grid_comm, &m_grid_comm);
      else
        m_grid_comm = MPI_COMM_SELF;
    }
    if (st.m_replica_comm != MPI_COMM_NULL) {
      if (st.m_replica_comm != MPI_COMM_SELF)
        MPI_Comm_dup(st.m_replica_comm, &m_replica_comm);
      else
        m_replica_comm = MPI_COMM_SELF;
    }
    if (st.m_plane_comm != MPI_COMM_NULL) {
      MPI_Comm_dup(st.m_plane_comm, &m_plane_comm);
    }
  }

  StateTBase(StateTBase&& st) noexcept
    : State()
    , ::hpg::runtime::StateT<D>(std::move(st)) {

    std::swap(m_reduced_grid, st.m_reduced_grid);
    std::swap(m_reduced_weights, st.m_reduced_weights);
    std::swap(m_vis_comm, st.m_vis_comm);
    std::swap(m_grid_comm, st.m_grid_comm);
    std::swap(m_replica_comm, st.m_replica_comm);
    std::swap(m_plane_comm, st.m_plane_comm);
  }

  virtual ~StateTBase() {}

  StateTBase&
  operator=(const StateTBase& st) {
    StateTBase tmp(st);
    this->swap(tmp);
    return *this;
  }

  StateTBase&
  operator=(StateTBase&& st) noexcept {
    StateTBase tmp(std::move(st));
    this->swap(tmp);
    return *this;
  }

  virtual unsigned
  degrid_execution_context(unsigned c) noexcept {
    return c;
  }

  virtual unsigned
  grid_execution_context(unsigned c) noexcept {
    return c;
  }

  void
  reduce_weights_unlocked() const {
    if (!m_reduced_weights) {
      if (non_trivial_visibility_partition()
          || non_trivial_replica_partition()) {
        auto is_root =
          is_visibility_partition_root() && is_replica_partition_root();
        fence();
        this->m_exec_contexts[0].switch_to_compute();
        MPI_Reduce(
          (is_visibility_partition_root()
           ? MPI_IN_PLACE
           : this->m_grid_weights.data()),
          this->m_grid_weights.data(),
          this->m_grid_weights.span(),
          mpi_datatype<grid_value_fp>::value(),
          MPI_SUM,
          0,
          m_vis_comm);
        if (is_visibility_partition_root())
          MPI_Reduce(
            (is_root ? MPI_IN_PLACE : this->m_grid_weights.data()),
            this->m_grid_weights.data(),
            this->m_grid_weights.span(),
            mpi_datatype<grid_value_fp>::value(),
            MPI_SUM,
            0,
            m_replica_comm);
        if (!is_root)
          const_cast<StateTBase<D>*>(this)->fill_grid_weights(grid_value_fp(0));
      }
      m_reduced_weights = true;
    }
  }

  void
  reduce_weights() const {
    std::scoped_lock lock(this->m_mtx);
    reduce_weights_unlocked();
  }

  void
  reduce_grid_unlocked() const {
    if (!m_reduced_grid) {
      if (non_trivial_visibility_partition()
          || non_trivial_replica_partition()) {
        auto is_root =
          is_visibility_partition_root() && is_replica_partition_root();
        fence();
        this->m_exec_contexts[0].switch_to_compute();
        MPI_Reduce(
          (is_visibility_partition_root() ? MPI_IN_PLACE : this->m_grid.data()),
          this->m_grid.data(),
          this->m_grid.span(),
          mpi_datatype<impl::gv_t>::value(),
          MPI_SUM,
          0,
          m_vis_comm);
        if (is_visibility_partition_root())
          MPI_Reduce(
            (is_root ? MPI_IN_PLACE : this->m_grid.data()),
            this->m_grid.data(),
            this->m_grid.span(),
            mpi_datatype<impl::gv_t>::value(),
            MPI_SUM,
            0,
            m_replica_comm);
        if (!is_root)
          const_cast<StateTBase<D>*>(this)->fill_grid(impl::gv_t(0));
      }
      m_reduced_grid = true;
    }
  }

  void
  reduce_grid() const {
    std::scoped_lock lock(this->m_mtx);
    reduce_grid_unlocked();
  }

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  set_convolution_function(
    unsigned context,
    Device host_device,
    CFArray&& cf_array) override {

    // N.B: access cf_array directly only at the root rank of m_grid_comm
    bool is_root = is_grid_partition_root();

    // Broadcast the cf_array in m_grid_comm, but as the equivalent of a
    // impl::DeviceCFArray for efficiency. Note that we broadcast the data among
    // host memories, which allows us to defer any device fence for as long as
    // possible.

    // format: vector of unsigned: (oversampling, <for each group> extent[0],
    // extent[1], extent[2], ...extent[CFArrayShape:rank - 1])
    std::vector<unsigned> shape;
    int shape_sz;
    using DevCFArray = typename impl::DeviceCFArray<D>;
    DevCFArray dev_cf_array;

    if (is_root) {
      try {
        dev_cf_array = dynamic_cast<DevCFArray&&>(cf_array);
      } catch (const std::bad_cast&) {
        dev_cf_array = DevCFArray(cf_array);
        for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp)
          impl::layout_for_device<D>(
            host_device,
            cf_array,
            grp,
            reinterpret_cast<CFArray::value_type*>(
              dev_cf_array.m_arrays[grp].data()));
      }
      shape.reserve(1 + dev_cf_array.num_groups() * CFArrayShape::rank);
      shape.push_back(dev_cf_array.oversampling());
      for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp)
        for (auto& e : dev_cf_array.extents(grp))
          shape.push_back(e);
      shape_sz = int(shape.size());
    }
    MPI_Bcast(
      &shape_sz,
      1,
      mpi_datatype<decltype(shape_sz)>::value(),
      0,
      m_grid_comm);
    shape.resize(shape_sz);
    MPI_Bcast(
      shape.data(),
      shape_sz,
      mpi_datatype<decltype(shape)::value_type>::value(),
      0,
      m_grid_comm);

    // initialize the dev_cf_array on non-root ranks
    if (!is_root)
      dev_cf_array = DevCFArray(DevCFShape(shape));

    // broadcast dev_cf_array values
    for (unsigned grp = 0; grp < dev_cf_array.num_groups(); ++grp)
      MPI_Bcast(
        dev_cf_array.m_arrays[grp].data(),
        dev_cf_array.m_arrays[grp].size(),
        mpi_datatype<CFArray::value_type>::value(),
        0,
        m_grid_comm);

    // all ranks now copy the CF kernels to device memory
    auto ctx = degrid_execution_context(context);
    this->m_exec_contexts[ctx].switch_to_copy(true);
    auto& cf = this->m_exec_contexts[ctx].current_cf_pool();
    cf->add_device_cfs(
      this->m_exec_contexts[ctx].current_exec_space(),
      std::move(dev_cf_array));

    return std::nullopt;
  }

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  set_model(Device host_device, GridValueArray&& gv) override {

    // N.B: access gv directly only at the root rank of m_vis_comm and only in
    // the root rank of any replicas
    bool is_root =
      is_visibility_partition_root() && is_replica_partition_root();

    // check that gv size equals that of grid
    {
      K::Array<int, 4> model_sz;
      if (is_root)
        model_sz = {
          int(gv.extent(0)),
          int(gv.extent(1)),
          int(gv.extent(2)),
          int(gv.extent(3))};
      if (is_visibility_partition_root())
        MPI_Bcast(
          model_sz.data(),
          4,
          mpi_datatype<decltype(model_sz)::value_type>::value(),
          0,
          m_replica_comm);
      MPI_Bcast(
        model_sz.data(),
        4,
        mpi_datatype<decltype(model_sz)::value_type>::value(),
        0,
        m_vis_comm);
      if (this->m_grid_size_local[0] != model_sz[0]
          || this->m_grid_size_local[1] != model_sz[1]
          || this->m_grid_size_local[2] != model_sz[2]
          || this->m_grid_size_local[3] != model_sz[3])
        return
          std::make_unique<InvalidModelGridSizeError>(
            model_sz,
            this->m_grid_size_local);
    }

    fence();
    this->m_exec_contexts.switch_to_copy();

    if (!this->m_model.is_allocated())
      this->m_model =
        decltype(this->m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          SD::grid_layout::dimensions(this->m_grid_size_local));

    // copy the model to device memory on the root rank, then broadcast it to
    // the other ranks
    if (is_root) {
      using GVVArray = typename impl::GridValueViewArray<D>;
      GVVArray gvv;
      try {
        gvv = dynamic_cast<GVVArray&&>(gv);
      } catch (const std::bad_cast&) {
        gvv.grid = K::create_mirror_view(this->m_model);
        switch (host_device) {
#ifdef HPG_ENABLE_SERIAL
        case Device::Serial:
          impl::init_model<Device::Serial>(gvv.grid, gv);
          break;
#endif
#ifdef HPG_ENABLE_OPENMP
        case Device::OpenMP:
          impl::init_model<Device::OpenMP>(gvv.grid, gv);
          break;
#endif
        default:
          assert(false);
          break;
        }
      }
      K::deep_copy(
        this->m_exec_contexts[0].current_exec_space(),
        this->m_model,
        gvv.grid);
      if (non_trivial_visibility_partition() || non_trivial_replica_partition())
        this->m_exec_contexts[0].current_exec_space().fence();
    }
    broadcast_model();

    return std::nullopt;
  }

  bool
  in_grid_channel_slice(int ch) const noexcept {
    constexpr auto ax = int(impl::core::GridAxis::channel);
    return
      (this->m_grid_offset_local[ax] <= ch)
      && (ch < (this->m_grid_offset_local[ax] + this->m_grid_size_local[ax]));
  }

  virtual std::unique_ptr<GridWeightArray>
  grid_weights() const override {
    std::scoped_lock lock(this->m_mtx);
    reduce_weights_unlocked();
    return
      ((is_visibility_partition_root() && is_replica_partition_root())
       ? this->grid_weights_unlocked()
       : nullptr);
  }

  virtual std::unique_ptr<GridValueArray>
  grid_values() const override {
    std::scoped_lock lock(this->m_mtx);
    reduce_grid_unlocked();
    return
      ((is_visibility_partition_root() && is_replica_partition_root())
       ? this->grid_values_unlocked()
       : nullptr);
  }

  virtual std::unique_ptr<GridValueArray>
  model_values() const override {
    return
      ((is_visibility_partition_root() && is_replica_partition_root())
       ? ::hpg::runtime::StateT<D>::model_values()
       : nullptr);
  }

  virtual void
  reset_grid() override {
    fence();
    this->new_grid(true, true);
    m_reduced_grid = true;
    m_reduced_weights = true;
  }

  virtual void
  normalize_by_weights(grid_value_fp wfactor) override {
    reduce_weights();
    reduce_grid();
    if (is_visibility_partition_root() && is_replica_partition_root())
      ::hpg::runtime::StateT<D>::normalize_by_weights(wfactor);
  }

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place) override {
    std::optional<std::unique_ptr<::hpg::Error>> err;
    reduce_grid();
    if (non_trivial_plane_partition()) {
      std::abort(); // FIXME
    } else if (is_visibility_partition_root() && is_replica_partition_root()) {
      err = ::hpg::runtime::StateT<D>::apply_grid_fft(norm, sign, in_place);
    }
    return err;
  }

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  apply_model_fft(grid_value_fp norm, FFTSign sign, bool in_place)
    override {
    std::optional<std::unique_ptr<::hpg::Error>> err;
    if (non_trivial_plane_partition()) {
      std::abort(); // FIXME
      broadcast_model();
    } else {
      err = ::hpg::runtime::StateT<D>::apply_model_fft(norm, sign, in_place);
    }
    return err;
  }

  virtual void
  shift_grid(ShiftDirection direction) override {
    reduce_grid();
    if (non_trivial_plane_partition()) {
      std::abort(); // FIXME
    } else if (is_visibility_partition_root() && is_replica_partition_root()) {
      ::hpg::runtime::StateT<D>::shift_grid(direction);
    }
  }

  virtual void
  shift_model(ShiftDirection direction) override {
    if (non_trivial_plane_partition()) {
      std::abort(); // FIXME
      broadcast_model();
    } else {
    ::hpg::runtime::StateT<D>::shift_model(direction);
    }
  }

protected:

  void
  swap(StateTBase& other) noexcept {
    ::hpg::runtime::StateT<D>::swap(
      static_cast<::hpg::runtime::StateT<D>&>(other));
    std::swap(m_vis_comm, other.m_vis_comm);
    std::swap(m_grid_comm, other.m_grid_comm);
    std::swap(m_replica_comm, other.m_replica_comm);
    std::swap(m_plane_comm, other.m_plane_comm);
    std::swap(m_reduced_grid, other.m_reduced_grid);
    std::swap(m_reduced_weights, other.m_reduced_weights);
  }

  void
  broadcast_model() noexcept {
    // broadcast the model values
    if (is_visibility_partition_root())
      MPI_Bcast(
        this->m_model.data(),
        this->m_model.span(),
        mpi_datatype<typename decltype(this->m_model)::value_type>::value(),
        0,
        m_replica_comm);
    MPI_Bcast(
      this->m_model.data(),
      this->m_model.span(),
      mpi_datatype<typename decltype(this->m_model)::value_type>::value(),
      0,
      m_vis_comm);
  }
};

template <Device D, VisibilityDistribution V>
class StateT
  : public StateTBase<D> {};

template <Device D>
class StateT<D, VisibilityDistribution::Broadcast>
  : public StateTBase<D> {
public:

  using ::hpg::runtime::StateT<D>::limit_tasks;

  using ::hpg::runtime::StateT<D>::repeated_value;

  using typename ::hpg::runtime::StateT<D>::maybe_vis_t;

  StateT(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    const ReplicatedGridBrick& grid_brick,
    MPI_Comm replica_comm,
    MPI_Comm plane_comm,
    unsigned num_contexts,
    unsigned max_active_tasks_per_context,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
  : StateTBase<D>(
    vis_comm,
    grid_comm,
    grid_brick,
    replica_comm,
    plane_comm,
    visibility_batch_size,
    max_avg_channels_per_vis,
    init_cf_shape,
    grid_size,
    grid_scale,
    mueller_indexes,
    conjugate_mueller_indexes,
    implementation_versions,
    ::hpg::runtime::ExecutionContextGroup<D>(
      mueller_indexes.m_npol,
      visibility_batch_size,
      max_avg_channels_per_vis * visibility_batch_size,
      repeated_value(
        num_contexts,
        limit_tasks(max_active_tasks_per_context)),
      num_contexts * limit_tasks(max_active_tasks_per_context))) {}

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

    // broadcast all vector values from the grid partition root rank
    std::array<MPI_Aint, 3> len;
    len[0] = visibilities.size();
    len[1] = wgt_values.size();
    assert(wgt_values.size() == wgt_col_index.size());
    len[2] = wgt_row_index.size();
    MPI_Bcast(len.data(), len.size(), MPI_AINT, 0, this->m_grid_comm);
    visibilities.resize(len[0]);
    {
      std::vector<MPI_Request> reqs;
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Ibcast(
        visibilities.data(),
        visibilities.size(),
        mpi_datatype<
          typename std::remove_reference_t<decltype(visibilities)>::value_type>
          ::value(),
        0,
        this->m_grid_comm,
        &reqs.back());
      wgt_values.resize(len[1]);
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Ibcast(
        wgt_values.data(),
        wgt_values.size(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_values)>::value_type>::value(),
        0,
        this->m_grid_comm,
        &reqs.back());
      wgt_col_index.resize(len[1]);
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Ibcast(
        wgt_col_index.data(),
        wgt_col_index.size(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_col_index)>::value_type>
          ::value(),
        0,
        this->m_grid_comm,
        &reqs.back());
      wgt_row_index.resize(len[2]);
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Ibcast(
        wgt_row_index.data(),
        wgt_row_index.size(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_row_index)>::value_type>
          ::value(),
        0,
        this->m_grid_comm,
        &reqs.back());
      MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    int replica_size;
    MPI_Comm_size(this->m_replica_comm, &replica_size);
    int replica_rank;
    MPI_Comm_rank(this->m_replica_comm, &replica_rank);

    // copy visibilities and channel mapping vectors to device
    this->m_exec_contexts[context].switch_to_copy(false);
    auto& sc_copy = this->m_exec_contexts[context].current_stream_context();
    sc_copy.copy_visibilities_to_device(std::move(visibilities));
    sc_copy.copy_weights_to_device(
      std::move(wgt_values),
      std::move(wgt_col_index),
      std::move(wgt_row_index));

    this->m_reduced_grid = this->m_reduced_grid && (len[0] == 0);
    this->m_reduced_weights =
      this->m_reduced_weights && (len[0] == 0 || !update_grid_weights);

    this->m_exec_contexts[context].switch_to_compute();
    auto& sc_grid = this->m_exec_contexts[context].current_stream_context();
    auto& cf = this->m_exec_contexts[context].current_cf_pool();
    auto gvis = sc_grid.template gvis_view<N>();
    using gvis0 =
      typename std::remove_reference_t<decltype(gvis)>::value_type;
    K::deep_copy(sc_grid.m_space, gvis, gvis0());

    // initialize the gridder object
    auto gridder =
      impl::core::VisibilityGridder(
        sc_grid.m_space,
        cf->cf_d,
        cf->cf_radii,
        cf->max_cf_extent_y,
        this->m_mueller_indexes,
        this->m_conjugate_mueller_indexes,
        len[0],
        replica_rank,
        replica_size,
        sc_grid.template visdata<N>(),
        sc_grid.m_weight_values.m_values,
        sc_grid.m_weight_col_index.m_values,
        sc_grid.m_weight_row_index.m_values,
        gvis,
        this->m_grid_scale,
        this->m_grid,
        this->m_grid_weights,
        this->m_model,
        this->m_grid_offset_local,
        this->m_grid_size_global);

    // use gridder object to invoke degridding and gridding kernels
    if (do_degrid) {
      gridder.degrid_all();
      // Whenever any visibilities are mapped to grid channels on multiple
      // ranks, we need to reduce the degridded visibility values from and to
      // all ranks. NB: this is a prime area for considering performance
      // improvements, but any solution should be scalable, and at least the
      // following satisfies that criterion
      sc_grid.fence();
      // Reduce all 4 polarizations, independent of the value of N, in order
      // to allow use of both a predefined datatype and a predefined reduction
      // operator, MPI_SUM. The function hpg::mpi::gvisbuff_datatype() could
      // be used to define a custom datatype, which would, for N < 4, be
      // non-contiguous and would also require a custom reduction
      // operator. The alternative might reduce the size of messages, but
      // would incur inefficiencies in the execution of the reduction
      // operation. A comparison between the alternatives might be worth
      // profiling, but for now, we go with the simpler approach.
      static_assert(
        sizeof(
          typename std::remove_reference_t<decltype(gvis)>::value_type)
        == N * sizeof(K::complex<visibility_fp>));
      MPI_Allreduce(
        MPI_IN_PLACE,
        gvis.data(),
        N * len[0],
        mpi_datatype<K::complex<visibility_fp>>::value(),
        MPI_SUM,
        this->m_grid_comm);
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
//     for (auto& [cube, supp] : *cf_indexes) {
//       auto& cfpool = std::get<0>(m_cfs[m_cf_indexes.front()]);
//       if ((supp >= cfpool.num_cf_groups)
//           || (cube >= cfpool.cf_d[supp].extent_int(5)))
//         return OutOfBoundsCFIndexError({cube, supp});
//     }
// #endif // NDEBUG

    switch (this->visibility_gridder_version()) {
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
    default:
      assert(false);
      std::abort();
      break;
    }
  }

  virtual std::variant<std::unique_ptr<::hpg::Error>, maybe_vis_t>
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
      return std::make_unique<::hpg::Error>("Assertion violation");
      break;
    }
  }
};

template <Device D>
class StateT<D, VisibilityDistribution::Pipeline>
  : public StateTBase<D> {
public:

  using ::hpg::runtime::StateT<D>::limit_tasks;

  using ::hpg::runtime::StateT<D>::repeated_value;

  using typename ::hpg::runtime::StateT<D>::maybe_vis_t;

  using StreamPhase = ::hpg::runtime::StreamPhase;

  std::array<int, 2> m_seqnums; // FIXME

  std::vector<MPI_Request> m_shift_requests;

  // TODO: can this algorithm be sensibly reduced in the case of a grid
  // partition size of one by unifying the degridding and gridding contexts?
  // Doing this would remove the pipeline delay, which exists currently even for
  // a pipeline size of one, reduce memory consumption, and become basically
  // equivalent to the sequential runtime gridder (in the grid partition
  // dimension). Some places in the code to consider this option are marked with
  // "***".

  StateT(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    const ReplicatedGridBrick& grid_brick,
    MPI_Comm replica_comm,
    MPI_Comm plane_comm,
    unsigned num_contexts_per_phase,
    unsigned max_active_tasks_per_context,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
  : StateTBase<D>(
    vis_comm,
    grid_comm,
    grid_brick,
    replica_comm,
    plane_comm,
    visibility_batch_size,
    max_avg_channels_per_vis,
    init_cf_shape,
    grid_size,
    grid_scale,
    mueller_indexes,
    conjugate_mueller_indexes,
    implementation_versions,
    ::hpg::runtime::ExecutionContextGroup<D>(
      mueller_indexes.m_npol,
      visibility_batch_size,
      max_avg_channels_per_vis * visibility_batch_size,
      repeated_value(
        2 * num_contexts_per_phase, // ***
        limit_tasks(max_active_tasks_per_context)),
      num_contexts_per_phase
        * (2 * limit_tasks(max_active_tasks_per_context) + 1))) {
  }

  virtual unsigned
  degrid_execution_context(unsigned c) noexcept override {
    return 2 * c; // ***
  }

  virtual unsigned
  grid_execution_context(unsigned c) noexcept override {
    return 2 * c + 1; // ***
  }

  template <typename S, typename V>
  void
  shift_context_data(S& src_d, S& dst_d, S& src_g, S& dst_g, V&& v) {
    // For this, we consider a logical pipeline of size 2 * size_gc, and all
    // ranks make two calls to MPI_Isend/MPI_Irecv pairs to move sets of
    // visibilities and weights around the logical pipeline. In the first call,
    // all ranks send data in the degrid_execution_context with the message tag
    // 0, and in the second call, all ranks send data in the
    // grid_execution_context with message tag 1. On the receive side, in the
    // first call, data are received in the degrid_execution_context with
    // message tag 0 on all ranks but the root, which receives message tag 1;
    // and in the second call, data are received in the grid_execution_context
    // with message tag 1 on all ranks but the root, which receives message tag
    // 0.
    int size_gc;
    MPI_Comm_size(this->m_grid_comm, &size_gc);
    int rank_gc;
    MPI_Comm_rank(this->m_grid_comm, &rank_gc);

    auto dest = (rank_gc + 1) % size_gc;
    auto source = (rank_gc + size_gc - 1) % size_gc;

    auto sendvect_d = v(src_d);
    auto recvvect_d = v(dst_d);
    auto sendvect_g = v(src_g);
    auto recvvect_g = v(dst_g);

    const std::array<int, 2> sendcounts{
      sendvect_d.m_values.extent_int(0),
      sendvect_g.m_values.extent_int(0)};
    std::array<int, 2> recvcounts;

    MPI_Sendrecv(
      sendcounts.data(),
      sendcounts.size(),
      MPI_INT,
      dest,
      -1,
      recvcounts.data(),
      recvcounts.size(),
      MPI_INT,
      source,
      -1,
      this->m_grid_comm,
      MPI_STATUS_IGNORE);

    recvvect_d.resize(size_t(recvcounts[0]));
    recvvect_g.resize(size_t(recvcounts[1]));

    auto& sendview_d = sendvect_d.m_values;
    auto& recvview_d = recvvect_d.m_values;
    auto& sendview_g = sendvect_g.m_values;
    auto& recvview_g = recvvect_g.m_values;

    const MPI_Datatype dt =
      mpi_datatype<
        typename std::remove_reference_t<decltype(sendview_d)>::value_type>
        ::value();

    // All source buffers are in compute phase, and destination buffers in the
    // following copy phase, but when there is only one stream in an execution
    // context, buffers in that context are identical and this implementation
    // fails.

    assert(sendview_d.data() != recvview_d.data());
    assert(sendview_g.data() != recvview_g.data());

    // Note that we have changed the order here of send/recv calls a bit from
    // the logical description at the top of this method.

    m_shift_requests.push_back(MPI_REQUEST_NULL);
    MPI_Irecv(
      recvview_d.data(),
      recvcounts[0],
      dt,
      source,
      ((rank_gc == 0) ? 1 : 0),
      this->m_grid_comm,
      &m_shift_requests.back());

    m_shift_requests.push_back(MPI_REQUEST_NULL);
    MPI_Irecv(
      recvview_g.data(),
      recvcounts[1],
      dt,
      source,
      ((rank_gc == 0) ? 0 : 1),
      this->m_grid_comm,
      &m_shift_requests.back());

    m_shift_requests.push_back(MPI_REQUEST_NULL);
    MPI_Isend(
      sendview_d.data(),
      sendcounts[0],
      dt,
      dest,
      0,
      this->m_grid_comm,
      &m_shift_requests.back());

    m_shift_requests.push_back(MPI_REQUEST_NULL);
    MPI_Isend(
      sendview_g.data(),
      sendcounts[1],
      dt,
      dest,
      1,
      this->m_grid_comm,
      &m_shift_requests.back());
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

    // TODO: for now, assume that the set of flags don't change in calls to this
    // method; ultimately, we could carry the flags with each set of
    // visibilities going around the pipeline, or just proclaim that it is
    // erroneous to change the flag values from one set of visibilities to the
    // next (in what scope?)

    // outline:
    // - at root rank in degrid_execution_context, copy new set of visibilities
    //   and weights to device (doing this first allows the "set CF", "grid vis"
    //   sequence to use a common execution context, like in the sequential
    //   runtime)
    // - set up VisibilityGridder in degrid_execution_context, do degridding
    // - also in degrid_execution_context, at tail rank, compute residual or
    //   predicted visibilities
    // - set up VisibilityGridder in grid_execution_context, do gridding
    // - shift visibilities and CFs in pipeline, both execution contexts
    // - at root rank in degrid_execution_context return residual/predicted
    //   visibilities

    int size_gc;
    MPI_Comm_size(this->m_grid_comm, &size_gc);
    int rank_gc;
    MPI_Comm_rank(this->m_grid_comm, &rank_gc);

    int size_rc;
    MPI_Comm_size(this->m_replica_comm, &size_rc);
    int rank_rc;
    MPI_Comm_rank(this->m_replica_comm, &rank_rc);

    maybe_vis_t result;

    // wait for completion of previous pipeline data shifts
    MPI_Waitall(
      m_shift_requests.size(),
      m_shift_requests.data(),
      MPI_STATUSES_IGNORE);
    m_shift_requests.clear();

    // copy new set of visibilities and weights to the device on root rank in
    // degrid_execution_context
    //
    if (rank_gc == 0) {
      auto& ctx =
        this->m_exec_contexts[this->degrid_execution_context(context)];
      ctx.switch_to_copy(false);
      auto& sc = ctx.current_stream_context();
      // a fence is needed here to complete the future of returned visibilities
      // from the previous call to this method. TODO: find a better approach?
      // The explicit fence on the StreamContext instance is needed because of
      // the way StreamContext completes the future returned by the previous
      // call to this method, so any change here will depend on a change to
      // StreamContext and how it deals with the future for the returned
      // visibilities.
      sc.fence();

      sc.copy_visibilities_to_device(std::move(visibilities));
      sc.copy_weights_to_device(
        std::move(wgt_values),
        std::move(wgt_col_index),
        std::move(wgt_row_index));
    }

    // do work in degridding context
    //
    {
      auto& ctx =
        this->m_exec_contexts[this->degrid_execution_context(context)];
      ctx.switch_to_compute();
      auto& sc = ctx.current_stream_context();
      auto& cf = ctx.current_cf_pool();

      auto gridder =
        impl::core::VisibilityGridder(
          sc.m_space,
          cf->cf_d,
          cf->cf_radii,
          cf->max_cf_extent_y,
          this->m_mueller_indexes,
          this->m_conjugate_mueller_indexes,
          sc.num_vis(),
          rank_rc,
          size_rc,
          sc.template visdata<N>(),
          sc.m_weight_values.m_values,
          sc.m_weight_col_index.m_values,
          sc.m_weight_row_index.m_values,
          sc.template gvis_view<N>(),
          this->m_grid_scale,
          this->m_grid,
          this->m_grid_weights,
          this->m_model,
          this->m_grid_offset_local,
          this->m_grid_size_global);

      // do degridding
      //
      if (do_degrid)
        gridder.degrid_all();

      // compute predicted or residual visibilities at tail rank
      //
      if (rank_gc == size_gc - 1) {
        if (do_degrid)
          if (do_grid)
            gridder.vis_copy_residual_and_rescale();
          else
            gridder.vis_copy_predicted();
        else
          gridder.vis_rescale();
      }
    }

    // do gridding (in gridding context)
    //
    if (do_grid) {
      auto& ctx =
        this->m_exec_contexts[this->grid_execution_context(context)];
      ctx.switch_to_compute();
      auto& sc = ctx.current_stream_context();
      auto& cf = ctx.current_cf_pool();

      auto num_vis = sc.num_vis();

      auto gridder =
        impl::core::VisibilityGridder(
          sc.m_space,
          cf->cf_d,
          cf->cf_radii,
          cf->max_cf_extent_y,
          this->m_mueller_indexes,
          this->m_conjugate_mueller_indexes,
          num_vis,
          rank_rc,
          size_rc,
          sc.template visdata<N>(),
          sc.m_weight_values.m_values,
          sc.m_weight_col_index.m_values,
          sc.m_weight_row_index.m_values,
          sc.template gvis_view<N>(),
          this->m_grid_scale,
          this->m_grid,
          this->m_grid_weights,
          this->m_model,
          this->m_grid_offset_local,
          this->m_grid_size_global);

      if (update_grid_weights)
        gridder.grid_all();
      else
        gridder.grid_all_no_weights();

      this->m_reduced_grid = this->m_reduced_grid && (num_vis == 0);
      this->m_reduced_weights =
        this->m_reduced_weights && (num_vis == 0 || !update_grid_weights);
    }

    // do pipeline shift of data (visibilities and weights) in both contexts
    //
    {
      auto& ctx_d =
        this->m_exec_contexts[this->degrid_execution_context(context)];
      ctx_d.switch_to_compute();
      auto& src_d = ctx_d.current_stream_context();
      ctx_d.switch_to_copy(false);
      auto& dst_d = ctx_d.current_stream_context();

      auto& ctx_g =
        this->m_exec_contexts[this->grid_execution_context(context)];
      ctx_g.switch_to_compute();
      auto& src_g = ctx_g.current_stream_context();
      ctx_g.switch_to_copy(false);
      auto& dst_g = ctx_g.current_stream_context();

      // FIXME: shift CFs

      shift_context_data(
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.template visibilities<N>(); });
      shift_context_data(
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.m_weight_values; });
      shift_context_data(
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.m_weight_col_index; });
      shift_context_data(
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.m_weight_row_index; });
      src_d.fence(); // degridding must complete before shifting gvis
      shift_context_data(
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.template gvis<N>(); });
    }

    // return visibilities at root rank in degridding execution context
    //
    if (rank_gc == 0) {
      auto& ctx =
        this->m_exec_contexts[this->degrid_execution_context(context)];
      auto& sc = ctx.current_stream_context();
      result = sc.copy_visibilities_to_host(return_visibilities);
    }

    return result;
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
//     for (auto& [cube, supp] : *cf_indexes) {
//       auto& cfpool = std::get<0>(m_cfs[m_cf_indexes.front()]);
//       if ((supp >= cfpool.num_cf_groups)
//           || (cube >= cfpool.cf_d[supp].extent_int(5)))
//         return OutOfBoundsCFIndexError({cube, supp});
//     }
// #endif // NDEBUG

    switch (this->visibility_gridder_version()) {
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
    default:
      assert(false);
      std::abort();
      break;
    }
  }

  virtual std::variant<std::unique_ptr<::hpg::Error>, maybe_vis_t>
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
      return std::make_unique<::hpg::Error>("Assertion violation");
      break;
    }
  }
};

} // end namespace hpg::mpi::runtime

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
