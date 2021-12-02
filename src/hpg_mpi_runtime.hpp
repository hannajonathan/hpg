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
  using value_type = void;
  static value_type value() {};
};
template <>
struct mpi_datatype<char> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CHAR; };
};
template <>
struct mpi_datatype<signed short int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_SHORT; }
};
template <>
struct mpi_datatype<signed int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_INT; }
};
template <>
struct mpi_datatype<signed long int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_LONG; }
};
template <>
struct mpi_datatype<signed long long int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_LONG_LONG; }
};
template <>
struct mpi_datatype<signed char> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_SIGNED_CHAR; }
};
template <>
struct mpi_datatype<unsigned char> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_CHAR; }
};
template <>
struct mpi_datatype<unsigned short int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_SHORT; }
};
template <>
struct mpi_datatype<unsigned int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED; }
};
template <>
struct mpi_datatype<unsigned long int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_LONG; }
};
template <>
struct mpi_datatype<unsigned long long int> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_UNSIGNED_LONG_LONG; }
};
template <>
struct mpi_datatype<float> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_FLOAT; }
};
template <>
struct mpi_datatype<double> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_DOUBLE; }
};
template <>
struct mpi_datatype<long double> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_LONG_DOUBLE; }
};
template <>
struct mpi_datatype<wchar_t> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_WCHAR; }
};
template <>
struct mpi_datatype<bool> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_BOOL; }
};
template <>
struct mpi_datatype<std::complex<float>> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_COMPLEX; }
};
template <>
struct mpi_datatype<std::complex<double>> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_DOUBLE_COMPLEX; }
};
template <>
struct mpi_datatype<std::complex<long double>> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_LONG_DOUBLE_COMPLEX; }
};
template <>
struct mpi_datatype<K::complex<float>> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_COMPLEX; }
};
template <>
struct mpi_datatype<K::complex<double>> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_DOUBLE_COMPLEX; }
};
template <>
struct mpi_datatype<K::complex<long double>> {
  using value_type = MPI_Datatype;
  static constexpr value_type value() { return MPI_CXX_LONG_DOUBLE_COMPLEX; }
};
template <unsigned N>
struct mpi_datatype<::hpg::VisData<N>> {
  using value_type = MPI_Datatype;
  static value_type
  value() {
    static std::once_flag flag;
    static MPI_Datatype result;
    std::call_once(
      flag,
      [](MPI_Datatype* dt) {
        using VD = ::hpg::VisData<N>;
        constexpr int count = 6;
        int blocklengths[count] = {
          VD::npol, // m_visibilities
          1, // m_frequency
          1, // m_phase
          std::tuple_size<decltype(VD::m_uvw)>::value,
          std::tuple_size<decltype(VD::m_cf_index)>::value,
          std::tuple_size<decltype(VD::m_cf_phase_gradient)>::value
        };
        MPI_Aint displacements[count] = {
          offsetof(VD, m_visibilities),
          offsetof(VD, m_frequency),
          offsetof(VD, m_phase),
          offsetof(VD, m_uvw),
          offsetof(VD, m_cf_index),
          offsetof(VD, m_cf_phase_gradient)
        };
        MPI_Datatype types[count] = {
          mpi_datatype<typename decltype(VD::m_visibilities)::value_type>
            ::value(),
          mpi_datatype<decltype(VD::m_frequency)>::value(),
          mpi_datatype<decltype(VD::m_phase)>::value(),
          mpi_datatype<typename decltype(VD::m_uvw)::value_type>::value(),
          mpi_datatype<typename decltype(VD::m_cf_index)::value_type>::value(),
          mpi_datatype<typename decltype(VD::m_cf_phase_gradient)::value_type>
            ::value()
        };
        MPI_Type_create_struct(
          count,
          blocklengths,
          displacements,
          types,
          dt);
        MPI_Type_commit(dt);
      },
      &result);
    return result;
  }
};
template <typename T, int N>
struct mpi_datatype<impl::core::poln_array_type<T, N>> {
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
  using value_type = MPI_Datatype;
  static value_type
  value() {
    static std::once_flag flag;
    static MPI_Datatype result;
    std::call_once(
      flag,
      [](MPI_Datatype* dt) {
        using VD = impl::core::VisData<N, V, F, P, U, G>;
        constexpr int count = 6;
        int blocklengths[count] = {
          VD::npol, // m_values
          1, // m_freq
          1, // m_d_phase
          decltype(VD::m_uvw)::size(),
          decltype(VD::m_cf_index)::size(),
          decltype(VD::m_cf_phase_gradient)::size()
        };
        MPI_Aint displacements[count] = {
          offsetof(VD, m_values),
          offsetof(VD, m_freq),
          offsetof(VD, m_d_phase),
          offsetof(VD, m_uvw),
          offsetof(VD, m_cf_index),
          offsetof(VD, m_cf_phase_gradient)
        };
        MPI_Datatype types[count] = {
          mpi_datatype<typename decltype(VD::m_values)::value_type>
            ::value(),
          mpi_datatype<decltype(VD::m_freq)>::value(),
          mpi_datatype<decltype(VD::m_d_phase)>::value(),
          mpi_datatype<typename decltype(VD::m_uvw)::value_type>::value(),
          mpi_datatype<typename decltype(VD::m_cf_index)::value_type>::value(),
          mpi_datatype<typename decltype(VD::m_cf_phase_gradient)::value_type>
            ::value()
        };
        MPI_Type_create_struct(
          count,
          blocklengths,
          displacements,
          types,
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

protected:

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
struct /*HPG_EXPORT*/ StateT
  : public ::hpg::runtime::StateT<D>
  , virtual public State {

  using typename ::hpg::runtime::StateT<D>::maybe_vis_t;

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

  StateT(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    const ReplicatedGridBrick& grid_brick,
    MPI_Comm replica_comm,
    MPI_Comm plane_comm,
    unsigned max_active_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
    : State(vis_comm, grid_comm, replica_comm, plane_comm)
    , ::hpg::runtime::StateT<D>(
      max_active_tasks,
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
      implementation_versions) {}

  StateT(const StateT& st)
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

  StateT(StateT&& st) noexcept
    : State()
    , ::hpg::runtime::StateT<D>(std::move(st)) {

    std::swap(m_reduced_grid, st.m_reduced_grid);
    std::swap(m_reduced_weights, st.m_reduced_weights);
    std::swap(m_vis_comm, st.m_vis_comm);
    std::swap(m_grid_comm, st.m_grid_comm);
    std::swap(m_replica_comm, st.m_replica_comm);
    std::swap(m_plane_comm, st.m_plane_comm);
  }

  virtual ~StateT() {}

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

  void
  reduce_weights_unlocked() const {
    if (!m_reduced_weights) {
      if (non_trivial_visibility_partition()
          || non_trivial_replica_partition()) {
        auto is_root =
          is_visibility_partition_root() && is_replica_partition_root();
        auto& exec =
          this->m_exec_spaces[
            this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
        exec.fence();
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
          const_cast<StateT<D>*>(this)->fill_grid_weights(grid_value_fp(0));
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
        auto& exec =
          this->m_exec_spaces[
            this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
        exec.fence();
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
          const_cast<StateT<D>*>(this)->fill_grid(impl::gv_t(0));
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
  set_convolution_function(Device host_device, CFArray&& cf_array) override {

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
      shape.reserve(1 + dev_cf_array.num_groups() * (CFArrayShape::rank - 1));
      shape.push_back(dev_cf_array.oversampling());
      for (unsigned grp = 0; grp < cf_array.num_groups(); ++grp)
        for (auto& e : dev_cf_array.extents(grp))
          shape.push_back(e);
      shape_sz = int(shape.size());
    }
    MPI_Bcast(&shape_sz, 1, mpi_datatype<int>::value(), 0, m_grid_comm);
    shape.resize(shape_sz);
    MPI_Bcast(
      shape.data(),
      shape_sz,
      mpi_datatype<unsigned>::value(),
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
    this->switch_cf_pool();
    auto& exec =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::PRE_GRIDDING)];
    auto& cf = std::get<0>(this->m_cfs[this->m_cf_indexes.front()]);
    cf.add_device_cfs(exec.space, std::move(dev_cf_array));

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
          mpi_datatype<unsigned>::value(),
          0,
          m_replica_comm);
      MPI_Bcast(
        model_sz.data(),
        4,
        mpi_datatype<unsigned>::value(),
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
    auto& exec =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::PRE_GRIDDING)];

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
      K::deep_copy(exec.space, this->m_model, gvv.grid);
      if (non_trivial_visibility_partition() || non_trivial_replica_partition())
        exec.fence();
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

  enum class GriddingAlgo {
    All2All,
    Pipeline
  };

  template <unsigned N>
  maybe_vis_t
  all2all_gridding(
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
    MPI_Bcast(len.data(), len.size(), MPI_AINT, 0, m_grid_comm);
    visibilities.resize(len[0]);
    {
      std::vector<MPI_Request> reqs;
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Ibcast(
        visibilities.data(),
        visibilities.size(),
        mpi_datatype<::hpg::VisData<N>>::value(),
        0,
        m_grid_comm,
        &reqs.back());
      wgt_values.resize(len[1]);
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Ibcast(
        wgt_values.data(),
        wgt_values.size(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_values)>::value_type>::value(),
        0,
        m_grid_comm,
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
        m_grid_comm,
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
        m_grid_comm,
        &reqs.back());
      MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    int replica_size;
    MPI_Comm_size(m_replica_comm, &replica_size);
    int replica_rank;
    MPI_Comm_rank(m_replica_comm, &replica_rank);

    // determine whether any visibility is being mapped to grid channels in
    // multiple (> 1) ranks of the grid partition (can be skipped if not
    // degridding)
    bool non_local_channel_mapping;
    if (non_trivial_grid_partition() && do_degrid) {
      std::vector<int> num_ranks_mapped(len[0]);
      // The following has the benefit of depending only on local data on every
      // rank, but it comes at the cost of a call to MPI_Allreduce();
      // alternatives may be considered if this design presents a performance
      // issue. In particular, at best we're avoiding one later call to
      // MPI_Allreduce after degridding but it comes at the expense of always
      // calling MPI_Allreduce here. Assuming that the post-degridding call to
      // MPI_Allreduce is slower than this one is reasonable, but not verified.

      // FIXME: should expand test to account for partitioned planes
      for (size_t i = replica_rank; i < len[0]; i += replica_size)
        num_ranks_mapped[i] =
          (std::any_of(
            &wgt_col_index[wgt_row_index[i]],
            &wgt_col_index[wgt_row_index[i + 1]],
            [this](auto& c) { return in_grid_channel_slice(c); })
           ? 1
           : 0);
      MPI_Allreduce(
        MPI_IN_PLACE,
        num_ranks_mapped.data(),
        len[0],
        mpi_datatype<decltype(num_ranks_mapped)::value_type>::value(),
        MPI_SUM,
        m_grid_comm);
      non_local_channel_mapping =
        std::any_of(
          num_ranks_mapped.begin(),
          num_ranks_mapped.end(),
          [](auto& n) { return n > 1; });
    } else {
      non_local_channel_mapping = false;
    }

    // copy visibilities and channel mapping vectors to device
    auto& exec_pre =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::PRE_GRIDDING)];
    exec_pre.copy_visibilities_to_device(std::move(visibilities));
    exec_pre.copy_weights_to_device(
      std::move(wgt_values),
      std::move(wgt_col_index),
      std::move(wgt_row_index));
    m_reduced_grid = m_reduced_grid && (len[0] == 0);
    m_reduced_weights =
      m_reduced_weights && (len[0] == 0 || !update_grid_weights);

    // initialize the gridder object
    auto& exec_grid =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::GRIDDING)];
    auto& cf = std::get<0>(this->m_cfs[this->m_cf_indexes.front()]);
    auto& gvisbuff = exec_grid.gvisbuff;
    K::deep_copy(
      exec_grid.space,
      gvisbuff,
      typename std::remove_reference_t<decltype(gvisbuff)>::value_type());

    auto gridder =
      impl::core::VisibilityGridder(
        exec_grid.space,
        cf.cf_d,
        cf.cf_radii,
        cf.max_cf_extent_y,
        this->m_mueller_indexes,
        this->m_conjugate_mueller_indexes,
        len[0],
        replica_rank,
        replica_size,
        exec_grid.template visdata<N>(),
        exec_grid.weight_values,
        exec_grid.weight_col_index,
        exec_grid.weight_row_index,
        gvisbuff,
        this->m_grid_scale,
        this->m_grid,
        this->m_grid_weights,
        this->m_model,
        this->m_grid_offset_local,
        this->m_grid_size_global);

    // use gridder object to invoke degridding and gridding kernels
    if (do_degrid) {
      gridder.degrid_all(len[0]);
      if (non_local_channel_mapping) {
        // Whenever any visibilities are mapped to grid channels on multiple
        // ranks, we need to reduce the degridded visibility values from and to
        // all ranks. NB: this is a prime area for considering performance
        // improvements, but any solution should be scalable, and at least the
        // following satisfies that criterion
        exec_grid.fence();
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
            typename std::remove_reference_t<decltype(gvisbuff)>::value_type)
          == 4 * sizeof(K::complex<visibility_fp>));
        MPI_Allreduce(
          MPI_IN_PLACE,
          gvisbuff.data(),
          4 * len[0],
          mpi_datatype<K::complex<visibility_fp>>::value(),
          MPI_SUM,
          m_grid_comm);
      }
      if (do_grid)
        gridder.vis_copy_residual_and_rescale(len[0]);
      else
        gridder.vis_copy_predicted(len[0]);
    } else {
      gridder.vis_rescale(len[0]);
    }

    if (do_grid) {
      if (update_grid_weights)
        gridder.grid_all(len[0]);
      else
        gridder.grid_all_no_weights(len[0]);
    }
    return exec_grid.copy_visibilities_to_host(return_visibilities);
  }

  template <unsigned N, typename F>
  void
  cycle_op(
    ::hpg::runtime::ExecSpace<D>& exec,
    int& num_vis,
    const int& max_vis,
    const int& max_wgts,
    F&& op) const {

    int sz;
    MPI_Comm_size(m_grid_comm, &sz);
    int rank;
    MPI_Comm_rank(m_grid_comm, &rank);
    int right = (rank + 1) % sz;
    int left = (rank + sz - 1) % sz;

    for (size_t i = 0; i < sz; ++i) {
      op(num_vis);

      exec.fence();

      // TODO: replace calls to MPI_Sendrecv_replace with MPI-4's
      // MPI_Isendrecv_replace when it's available
      // std::vector<MPI_Request> reqs;
      // reqs.push_back(MPI_REQUEST_NULL);
      MPI_Sendrecv_replace(
        &num_vis,
        1,
        MPI_INT,
        left,
        0,
        right,
        0,
        m_grid_comm,
        MPI_STATUS_IGNORE);
      // reqs.push_back(MPI_REQUEST_NULL);
      MPI_Sendrecv_replace(
        exec.weight_values.data(),
        max_wgts,
        mpi_datatype<vis_weight_fp>::value(),
        left,
        0,
        right,
        0,
        m_grid_comm,
        MPI_STATUS_IGNORE);
      // reqs.push_back(MPI_REQUEST_NULL);
      MPI_Sendrecv_replace(
        exec.weight_col_index.data(),
        max_wgts,
        mpi_datatype<unsigned>::value(),
        left,
        0,
        right,
        0,
        m_grid_comm,
        MPI_STATUS_IGNORE);
      // reqs.push_back(MPI_REQUEST_NULL);
      MPI_Sendrecv_replace(
        exec.weight_row_index.data(),
        max_vis + 1,
        mpi_datatype<size_t>::value(),
        left,
        0,
        right,
        0,
        m_grid_comm,
        MPI_STATUS_IGNORE);
      // reqs.push_back(MPI_REQUEST_NULL);
      auto visdata = exec.template visdata<N>();
      MPI_Sendrecv_replace(
        visdata.data(),
        max_vis,
        mpi_datatype<typename decltype(visdata)::value_type>::value(),
        left,
        0,
        right,
        0,
        m_grid_comm,
        MPI_STATUS_IGNORE);
      // reqs.push_back(MPI_REQUEST_NULL);
      MPI_Sendrecv_replace(
        exec.gvisbuff.data(),
        max_vis,
        mpi_datatype<typename decltype(exec.gvisbuff)::value_type>::value(),
        left,
        0,
        right,
        0,
        m_grid_comm,
        MPI_STATUS_IGNORE);
      // MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }
  }

  template <unsigned N>
  maybe_vis_t
  pipeline_gridding(
    std::vector<::hpg::VisData<N>>&& visibilities,
    std::vector<vis_weight_fp>&& wgt_values,
    std::vector<unsigned>&& wgt_col_index,
    std::vector<size_t>&& wgt_row_index,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

    // broadcast info for number of visibilities per rank
    int gsz;
    MPI_Comm_size(m_grid_comm, &gsz);
    int grnk;
    MPI_Comm_rank(m_grid_comm, &grnk);
    std::array<int, 5> vszs;
    vszs[0] = visibilities.size() / gsz; // min num vis per rank
    vszs[1] = visibilities.size() % gsz; // remainder of num vis over ranks
    vszs[2] = m_reduced_grid && visibilities.empty();
    vszs[3] =
      m_reduced_weights && (visibilities.empty() || !update_grid_weights);
    vszs[4] = 0;
    for (size_t i = 0; i < wgt_row_index.size(); ++i)
      vszs[4] =
        std::max(vszs[4], int(wgt_row_index[i + 1] - wgt_row_index[i]));
    MPI_Bcast(vszs.data(), vszs.size(), MPI_INT, 0, m_grid_comm);
    auto& min_vis = vszs[0];
    auto& rem_vis = vszs[1];
    auto max_vis = min_vis + ((rem_vis > 0) ? 1 : 0);
    m_reduced_grid = vszs[2];
    m_reduced_weights = vszs[3];
    auto& max_wgts = vszs[4];
    // num vis at this rank
    int num_vis = min_vis + ((grnk < rem_vis) ? 1 : 0);

    // scatter input data to all ranks
    {
      std::vector<MPI_Request> reqs;


      // scatter share of visibilities to all ranks
      std::vector<int> vis_sendcounts;
      vis_sendcounts.reserve(gsz);
      std::vector<int> vis_displs;
      vis_displs.reserve(gsz);
      for (int i = 0; i < gsz; ++i) {
        vis_sendcounts.push_back(min_vis + ((i < rem_vis) ? 1 : 0));
        vis_displs.push_back(i * min_vis + std::min(i, rem_vis));
      }
      // increase size of visibilities vector on non-root ranks
      visibilities.resize(std::max(visibilities.size(), size_t(max_vis)));
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Iscatterv(
        visibilities.data(),
        vis_sendcounts.data(),
        vis_displs.data(),
        mpi_datatype<::hpg::VisData<N>>::value(),
        is_grid_partition_root() ? MPI_IN_PLACE : visibilities.data(),
        num_vis,
        mpi_datatype<::hpg::VisData<N>>::value(),
        0,
        m_grid_comm,
        &reqs.back());

      // scatter share of wgt_row_index values to all ranks according to local
      // set of visibilities
      std::vector<int> wri_sendcounts;
      wri_sendcounts.reserve(gsz);
      std::vector<int> wri_displs;
      wri_displs.reserve(gsz);
      for (int i = 0; i < gsz; ++i) {
        wri_sendcounts.push_back(min_vis + 1 + ((i < rem_vis) ? 1 : 0));
        wri_displs.push_back(i * min_vis + std::min(i, rem_vis));
      }
      // increase size of wgt_row_index vector on non-root ranks
      wgt_row_index.resize(std::max(wgt_row_index.size(), size_t(max_vis + 1)));
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Iscatterv(
        wgt_row_index.data(),
        wri_sendcounts.data(),
        wri_displs.data(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_row_index)>::value_type>
          ::value(),
        is_grid_partition_root() ? MPI_IN_PLACE : wgt_row_index.data(),
        num_vis + 1,
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_row_index)>::value_type>
          ::value(),
        0,
        m_grid_comm,
        &reqs.back());
      // adjust wgt_row_index values to reflect local arrays
      for (int i = 0; i <= num_vis; ++i)
        wgt_row_index[i] -= wgt_row_index[0];

      // scatter share of wgt_col_index and wgt_values to all ranks according to
      // local set of visibilities
      int num_wgt_values = int(wgt_row_index[num_vis]);
      // increase size of wgt_col_index and wgt_values on non-root ranks
      wgt_col_index.resize(std::max(int(wgt_col_index.size()), max_wgts));
      wgt_values.resize(std::max(int(wgt_values.size()), max_wgts));
      std::vector<int> wv_sendcounts;
      wv_sendcounts.reserve(gsz);
      std::vector<int> wv_displs;
      wv_displs.reserve(gsz);
      for (int i = 0; i < gsz; ++i) {
        auto begin_wgt_row_idx =
          wgt_row_index[i * min_vis + std::min(i, rem_vis)];
        auto end_wgt_row_idx =
          wgt_row_index[(i + 1) * min_vis + std::min(i + 1, rem_vis)];
        wv_sendcounts.push_back(end_wgt_row_idx - begin_wgt_row_idx);
        wv_displs.push_back(begin_wgt_row_idx);
      }
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Iscatterv(
        wgt_col_index.data(),
        wv_sendcounts.data(),
        wv_displs.data(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_col_index)>::value_type>
          ::value(),
        is_grid_partition_root() ? MPI_IN_PLACE : wgt_col_index.data(),
        num_wgt_values,
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_col_index)>::value_type>
          ::value(),
        0,
        m_grid_comm,
        &reqs.back());
      reqs.push_back(MPI_REQUEST_NULL);
      MPI_Iscatterv(
        wgt_values.data(),
        wv_sendcounts.data(),
        wv_displs.data(),
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_values)>::value_type>::value(),
        is_grid_partition_root() ? MPI_IN_PLACE : wgt_values.data(),
        num_wgt_values,
        mpi_datatype<
          std::remove_reference_t<decltype(wgt_values)>::value_type>::value(),
        0,
        m_grid_comm,
        &reqs.back());
      MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

      // resize input vectors...actually reduces them only on rank 0
      visibilities.resize(max_vis);
      wgt_values.resize(max_wgts);
      wgt_col_index.resize(max_wgts);
      wgt_row_index.resize(max_vis + 1);
    }

    int replica_size;
    MPI_Comm_size(m_replica_comm, &replica_size);
    int replica_rank;
    MPI_Comm_rank(m_replica_comm, &replica_rank);

    // copy visibilities and channel mapping vectors to device
    auto& exec_pre =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::PRE_GRIDDING)];
    exec_pre.copy_visibilities_to_device(std::move(visibilities));
    exec_pre.copy_weights_to_device(
      std::move(wgt_values),
      std::move(wgt_col_index),
      std::move(wgt_row_index));

    // initialize the gridder object
    auto& exec_grid =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::GRIDDING)];
    auto& cf = std::get<0>(this->m_cfs[this->m_cf_indexes.front()]);
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
        this->m_mueller_indexes,
        this->m_conjugate_mueller_indexes,
        max_vis,
        replica_rank,
        replica_size,
        exec_grid.template visdata<N>(),
        exec_grid.weight_values,
        exec_grid.weight_col_index,
        exec_grid.weight_row_index,
        gvisbuff,
        this->m_grid_scale,
        this->m_grid,
        this->m_grid_weights,
        this->m_model,
        this->m_grid_offset_local,
        this->m_grid_size_global);

    // use gridder object to invoke degridding and gridding kernels
    if (do_degrid) {
      cycle_op<N>(
        exec_grid,
        num_vis,
        max_vis,
        max_wgts,
        [&](int& nv){ gridder.degrid_all(nv); });
      if (do_grid)
        gridder.vis_copy_residual_and_rescale(num_vis);
      else
        gridder.vis_copy_predicted(num_vis);
    } else {
      gridder.vis_rescale(num_vis);
    }

    if (do_grid) {
      if (update_grid_weights)
        cycle_op<N>(
          exec_grid,
          num_vis,
          max_vis,
          max_wgts,
          [&](int& nv) { gridder.grid_all(nv); });
      else
        cycle_op<N>(
          exec_grid,
          num_vis,
          max_vis,
          max_wgts,
          [&](int& nv) { gridder.grid_all_no_weights(nv); });
    }
    // FIXME: reassemble (gather) returned visibilities
    return exec_grid.copy_visibilities_to_host(return_visibilities);
  }

  template <unsigned N>
  maybe_vis_t
  default_grid_visibilities(
    Device /*host_device*/,
    std::vector<::hpg::VisData<N>>&& visibilities,
    std::vector<vis_weight_fp>&& wgt_values,
    std::vector<unsigned>&& wgt_col_index,
    std::vector<size_t>&& wgt_row_index,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

    int sz;
    MPI_Comm_size(m_grid_comm, &sz);
    switch (select_gridding_algorithm(
              visibilities.size(),
              wgt_values.size(),
              update_grid_weights,
              do_degrid,
              return_visibilities,
              do_grid)) {
    case GriddingAlgo::All2All:
      return
        all2all_gridding(
          std::move(visibilities),
          std::move(wgt_values),
          std::move(wgt_col_index),
          std::move(wgt_row_index),
          update_grid_weights,
          do_degrid,
          return_visibilities,
          do_grid);
      break;
    case GriddingAlgo::Pipeline:
      return
        pipeline_gridding(
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
      return maybe_vis_t();
    }
  }

  template <unsigned N>
  maybe_vis_t
  grid_visibilities(
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

    switch (visibility_gridder_version()) {
    case 0:
      return
        default_grid_visibilities(
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
  swap(StateT& other) noexcept {
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
        mpi_datatype<impl::gv_t>::value(),
        0,
        m_replica_comm);
    MPI_Bcast(
      this->m_model.data(),
      this->m_model.span(),
      mpi_datatype<impl::gv_t>::value(),
      0,
      m_vis_comm);
  }

  GriddingAlgo
  select_gridding_algorithm(
    size_t /*num_vis*/,
    size_t /*num_ch*/,
    bool /*update_grid_weights*/,
    bool /*do_degrid*/,
    bool /*return_visibilities*/,
    bool /*do_grid*/) const {
    return GriddingAlgo::All2All;
  }
};

} // end namespace hpg::mpi::runtime

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
