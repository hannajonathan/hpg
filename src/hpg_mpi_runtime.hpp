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

union GriddingFlags {
  uint8_t u;
  struct {
    bool update_grid_weights: 1;
    bool do_degrid: 1;
    bool return_visibilities: 1;
    bool do_grid: 1;
    bool fence: 1;
  } b;
};

MPI_Comm
dup_comm(MPI_Comm comm) {
  MPI_Comm result;
  if (comm == MPI_COMM_NULL)
    result = comm;
  else
    MPI_Comm_dup(comm, &result);
  return result;
}

struct ReplicatedGridBrick {
  // three axes are x, y, and channel; mrow partition not supported
  static constexpr unsigned rank = 3;
  enum Axis {x, y, channel};

  unsigned num_replicas;
  std::array<unsigned, rank> offset;
  std::array<unsigned, rank> size;
};

struct /*HPG_EXPORT*/ State
  : virtual public ::hpg::runtime::State {

protected:

  // communicator for subspace of visibility partition for this element of grid
  // partition
  MPI_Comm m_vis_comm;

  // Degridding and gridding communicators for subspace of grid plane partition
  // for this element of visibility partition. Two cases:
  // - no split phase: all communicators are identical
  // - with split phase: m_gp_comm is an intra-communicator, another one of
  //                     other two is an intra-communicator, and the third
  //                     is an inter-communicator
  MPI_Comm m_gp_comm; // comm over all of grid partition
  MPI_Comm m_degrid_comm; // comm over degridding sub-partition
  MPI_Comm m_grid_comm; // comm over gridding sub-partition

  MPI_Comm m_replica_comm;

  MPI_Comm m_plane_comm;

  mutable bool m_reduced_grid;

  mutable bool m_reduced_weights;

  State()
    : m_vis_comm(MPI_COMM_NULL)
    , m_gp_comm(MPI_COMM_NULL)
    , m_degrid_comm(MPI_COMM_NULL)
    , m_grid_comm(MPI_COMM_NULL)
    , m_replica_comm(MPI_COMM_NULL)
    , m_plane_comm(MPI_COMM_NULL)
    , m_reduced_grid(true)
    , m_reduced_weights(true) {}

public:

  State(
    MPI_Comm vis_comm,
    MPI_Comm gp_comm,
    MPI_Comm degrid_comm,
    MPI_Comm grid_comm,
    MPI_Comm replica_comm,
    MPI_Comm plane_comm)
    : m_vis_comm(vis_comm)
    , m_gp_comm(gp_comm)
    , m_degrid_comm(degrid_comm)
    , m_grid_comm(grid_comm)
    , m_replica_comm(replica_comm)
    , m_plane_comm(plane_comm)
    , m_reduced_grid(true)
    , m_reduced_weights(true) {}

  virtual ~State() {
    if (is_split_phase()) {
      MPI_Comm_free(&m_degrid_comm);
      MPI_Comm_free(&m_grid_comm);
    }
    for (auto& c :
           {&m_vis_comm, &m_gp_comm, &m_replica_comm, &m_plane_comm}) {
      if (*c != MPI_COMM_NULL && *c != MPI_COMM_WORLD)
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
  non_trivial_gpart() const noexcept {
    int size;
    MPI_Comm_size(m_gp_comm, &size);
    return size > 1;
  }

  bool
  is_gpart_root() const noexcept {
    int rank;
    MPI_Comm_rank(m_gp_comm, &rank);
    return rank == 0;
  }

  bool
  is_split_phase() const noexcept {
    return
      m_gp_comm != MPI_COMM_NULL
      && m_grid_comm != m_degrid_comm;
  }

  bool
  is_degridding() const noexcept {
    int flag;
    MPI_Comm_test_inter(m_degrid_comm, &flag);
    return !bool(flag);
  }

  bool
  is_gridding() const noexcept {
    int flag;
    MPI_Comm_test_inter(m_grid_comm, &flag);
    return !bool(flag);
  }

  bool
  non_trivial_degridding_partition() const noexcept {
    if (!is_degridding())
      return false;
    int size;
    MPI_Comm_size(m_degrid_comm, &size);
    return size > 1;
  }

  bool
  non_trivial_gridding_partition() const noexcept {
    if (!is_gridding())
      return false;
    int size;
    MPI_Comm_size(m_grid_comm, &size);
    return size > 1;
  }

  bool
  is_degridding_partition_root() const noexcept {
    if (!is_degridding())
      return false;
    int rank;
    MPI_Comm_rank(m_degrid_comm, &rank);
    return rank == 0;
  }

  bool
  is_gridding_partition_root() const noexcept {
    if (!is_gridding())
      return false;
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
    MPI_Comm gp_comm,
    MPI_Comm degrid_comm,
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
    : State(vis_comm, gp_comm, degrid_comm, grid_comm, replica_comm, plane_comm)
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

    dup_comm(st.m_vis_comm, &m_vis_comm);
    dup_comm(st.m_gp_comm, &m_gp_comm);
    if (!st.split_phase()) {
      m_grid_comm = m_gp_comm;
      m_degrid_comm = m_gp_comm;
    } else {
      dup_comm(st.m_degrid_comm, &m_degrid_comm);
      dup_comm(st.m_grid_comm, &m_grid_comm);
    }
    dup_comm(st.m_replica_comm, &m_replica_comm);
    dup_comm(st.m_plane_comm, &m_plane_comm);
  }

  StateTBase(StateTBase&& st) noexcept
    : State()
    , ::hpg::runtime::StateT<D>(std::move(st)) {

    std::swap(m_reduced_grid, st.m_reduced_grid);
    std::swap(m_reduced_weights, st.m_reduced_weights);
    std::swap(m_vis_comm, st.m_vis_comm);
    std::swap(m_gp_comm, st.m_gp_comm);
    std::swap(m_degrid_comm, st.m_degrid_comm);
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
  degrid_execution_context(unsigned c) const noexcept {
    return c;
  }

  virtual unsigned
  grid_execution_context(unsigned c) const noexcept {
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
    std::swap(m_gp_comm, other.m_gp_comm);
    std::swap(m_degrid_comm, other.m_degrid_comm);
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
    MPI_Comm degrid_comm,
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
    degrid_comm,
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

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  set_convolution_function(
    unsigned context,
    Device host_device,
    CFArray&& cf_array) override {

    // N.B: access cf_array directly only at the root rank of m_grid_comm
    bool is_root = this->is_grid_partition_root();

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
      this->m_grid_comm);
    shape.resize(shape_sz);
    MPI_Bcast(
      shape.data(),
      shape_sz,
      mpi_datatype<decltype(shape)::value_type>::value(),
      0,
      this->m_grid_comm);

    // initialize the dev_cf_array on non-root ranks
    if (!is_root)
      dev_cf_array = DevCFArray(::hpg::runtime::DevCFShape(shape));

    // broadcast dev_cf_array values
    for (unsigned grp = 0; grp < dev_cf_array.num_groups(); ++grp)
      MPI_Bcast(
        dev_cf_array.m_arrays[grp].data(),
        dev_cf_array.m_arrays[grp].size(),
        mpi_datatype<CFArray::value_type>::value(),
        0,
        this->m_grid_comm);

    // all ranks now copy the CF kernels to device memory
    auto& ctx = this->m_exec_contexts[this->degrid_execution_context(context)];
    ctx.switch_to_copy(true);
    auto& cf = ctx.current_cf_pool();
    cf->add_device_cfs(ctx.current_exec_space(), std::move(dev_cf_array));

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
protected:

public:

  using ::hpg::runtime::StateT<D>::limit_tasks;

  using ::hpg::runtime::StateT<D>::repeated_value;

  using typename ::hpg::runtime::StateT<D>::maybe_vis_t;

  using StreamPhase = ::hpg::runtime::StreamPhase;

protected:

  mutable std::vector<std::vector<MPI_Request>> m_shift_requests;

  mutable std::vector<bool> m_pipeline_flush;

public:

  // TODO: can this algorithm be sensibly reduced in the case of a grid
  // partition size of one by unifying the degridding and gridding contexts?
  // Doing this would remove the pipeline delay, which exists currently even for
  // a pipeline size of one, reduce memory consumption, and become basically
  // equivalent to the sequential runtime gridder (in the grid partition
  // dimension). Some places in the code to consider this option are marked with
  // "***".

  StateT(
    MPI_Comm vis_comm,
    MPI_Comm degrid_comm,
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
    degrid_comm,
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

    this->m_exec_contexts.for_all_stream_contexts(
      [](auto& sc) {
        sc.m_user_data = GriddingFlags();
      });
    m_shift_requests.resize(num_contexts_per_phase);
    m_pipeline_flush.resize(num_contexts_per_phase);
  }

  virtual unsigned
  degrid_execution_context(unsigned c) const noexcept override {
    return 2 * c; // ***
  }

  virtual unsigned
  grid_execution_context(unsigned c) const noexcept override {
    return 2 * c + 1; // ***
  }

private:

  MPI_Request*
  add_request(std::vector<MPI_Request>& reqs) const {
    reqs.push_back(MPI_REQUEST_NULL);
    return &reqs.back();
  }

protected:

  template <typename V>
  void
  shift_context_data(
    unsigned context,
    ::hpg::runtime::StreamContext<D>& src_d,
    ::hpg::runtime::StreamContext<D>& dst_d,
    ::hpg::runtime::StreamContext<D>& src_g,
    ::hpg::runtime::StreamContext<D>& dst_g,
    V&& v) const {
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

    if (recvcounts[0] > 0)
      MPI_Irecv(
        recvview_d.data(),
        recvcounts[0],
        dt,
        source,
        ((rank_gc == 0) ? 1 : 0),
        this->m_grid_comm,
        add_request(m_shift_requests[context]));

    if (recvcounts[1] > 0)
      MPI_Irecv(
        recvview_g.data(),
        recvcounts[1],
        dt,
        source,
        ((rank_gc == 0) ? 0 : 1),
        this->m_grid_comm,
        add_request(m_shift_requests[context]));

    if (sendcounts[0] > 0)
      MPI_Isend(
        sendview_d.data(),
        sendcounts[0],
        dt,
        dest,
        0,
        this->m_grid_comm,
        add_request(m_shift_requests[context]));

    if (sendcounts[1] > 0)
      MPI_Isend(
        sendview_g.data(),
        sendcounts[1],
        dt,
        dest,
        1,
        this->m_grid_comm,
        add_request(m_shift_requests[context]));
  }

  void
  shift_cfs(
    unsigned context,
    ::hpg::runtime::ExecutionContext<D>& ctx_d,
    ::hpg::runtime::ExecutionContext<D>& ctx_g) const {
    // NB: this eventually calls switch_to_copy() on each context

    int size_gc;
    MPI_Comm_size(this->m_grid_comm, &size_gc);
    int rank_gc;
    MPI_Comm_rank(this->m_grid_comm, &rank_gc);

    auto dest = (rank_gc + 1) % size_gc;
    auto source = (rank_gc + size_gc - 1) % size_gc;

    std::array<typename ::hpg::runtime::CFPoolRepo<D>::id_type, 2> cf_ids{
      ctx_d.current_cf_pool_id(),
      ctx_g.current_cf_pool_id()};
    decltype(cf_ids) next_cf_ids;
    {
      MPI_Datatype dt =
        mpi_datatype<typename decltype(cf_ids)::value_type>::value();
      MPI_Request req;
      MPI_Irecv(
        next_cf_ids.data(),
        next_cf_ids.size(),
        dt,
        source,
        0,
        this->m_grid_comm, &req);
      MPI_Send(cf_ids.data(), cf_ids.size(), dt, dest, 0, this->m_grid_comm);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    if (rank_gc == 0) // rank 0 never receives an upstream CF for degridding
      next_cf_ids[0] == cf_ids[0];
    std::array<bool, 2> do_send;
    std::array<bool, 2> do_recv{
      cf_ids[0] != next_cf_ids[0] && !ctx_d.find_pool(next_cf_ids[0]),
      cf_ids[1] != next_cf_ids[1] && !ctx_g.find_pool(next_cf_ids[1])};
    {
      MPI_Datatype dt =
        mpi_datatype<typename decltype(do_send)::value_type>::value();
      MPI_Request req;
      MPI_Irecv(
        do_send.data(),
        do_send.size(),
        dt,
        dest,
        0,
        this->m_grid_comm,
        &req);
      MPI_Send(do_recv.data(), do_recv.size(), dt, source, 0, this->m_grid_comm);
      MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    // exchange CF metadata
    auto src_cf_d = ctx_d.current_cf_pool();
    auto src_cf_g = ctx_g.current_cf_pool();
    ctx_d.switch_to_copy(cf_ids[0] != next_cf_ids[0], next_cf_ids[0]);
    ctx_g.switch_to_copy(cf_ids[1] != next_cf_ids[1], next_cf_ids[1]);
    auto dst_cf_d = ctx_d.current_cf_pool();
    auto dst_cf_g = ctx_g.current_cf_pool();
    // num groups
    unsigned dst_num_groups_d = 0, dst_num_groups_g = 0;
    {
      std::vector<MPI_Request> reqs;
      if (do_send[0])
        MPI_Isend(
          &src_cf_d->num_cf_groups,
          1,
          mpi_datatype<
            typename std::remove_reference_t<decltype(src_cf_d->num_cf_groups)>>
            ::value(),
          dest,
          0,
          this->m_grid_comm,
          add_request(reqs));
      if (do_send[1])
        MPI_Isend(
          &src_cf_g->num_cf_groups,
          1,
          mpi_datatype<
            typename std::remove_reference_t<decltype(src_cf_g->num_cf_groups)>>
            ::value(),
          dest,
          0,
          this->m_grid_comm,
          add_request(reqs));
      if (do_recv[0])
        MPI_Irecv(
          &dst_num_groups_d,
          1,
          mpi_datatype<decltype(dst_num_groups_d)>::value(),
          source,
          0,
          this->m_grid_comm,
          add_request(reqs));
      if (do_recv[1])
        MPI_Irecv(
          &dst_num_groups_g,
          1,
          mpi_datatype<decltype(dst_num_groups_g)>::value(),
          source,
          0,
          this->m_grid_comm,
          add_request(reqs));
      MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }
    // shape
    {
      ::hpg::runtime::DevCFShape src_shape_d;
      ::hpg::runtime::DevCFShape src_shape_g;
      ::hpg::runtime::DevCFShape dst_shape_d;
      ::hpg::runtime::DevCFShape dst_shape_g;
      std::vector<MPI_Request> reqs;
      if (do_send[0]) {
        src_shape_d = src_cf_d->shape();
        MPI_Isend(
          src_shape_d.m_shape.data(),
          src_shape_d.m_shape.size(),
          mpi_datatype<typename decltype(src_shape_d.m_shape)::value_type>
            ::value(),
          dest,
          0,
          this->m_grid_comm,
          add_request(reqs));
      }
      if (do_send[1]) {
        src_shape_g = src_cf_g->shape();
        MPI_Isend(
          src_shape_g.m_shape.data(),
          src_shape_g.m_shape.size(),
          mpi_datatype<typename decltype(src_shape_g.m_shape)::value_type>
            ::value(),
          dest,
          0,
          this->m_grid_comm,
          add_request(reqs));
      }
      if (do_recv[0]) {
        dst_shape_d = ::hpg::runtime::DevCFShape(dst_num_groups_d);
        MPI_Irecv(
          dst_shape_d.m_shape.data(),
          dst_shape_d.m_shape.size(),
          mpi_datatype<typename decltype(dst_shape_d.m_shape)::value_type>
            ::value(),
          source,
          0,
          this->m_grid_comm,
          add_request(reqs));
      }
      if (do_recv[1]) {
        dst_shape_g = ::hpg::runtime::DevCFShape(dst_num_groups_g);
        MPI_Irecv(
          dst_shape_g.m_shape.data(),
          dst_shape_g.m_shape.size(),
          mpi_datatype<typename decltype(dst_shape_g.m_shape)::value_type>
            ::value(),
          source,
          0,
          this->m_grid_comm,
          add_request(reqs));
      }
      MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

      if (do_recv[0])
        dst_cf_d->set_shape(dst_shape_d);
      if (do_recv[1])
        dst_cf_g->set_shape(dst_shape_g);
    }

    // exchange CF values
    if (do_send[0])
      MPI_Isend(
        src_cf_d->pool.data(),
        src_cf_d->size,
        mpi_datatype<impl::cf_t>::value(),
        dest,
        0,
        this->m_grid_comm,
        add_request(m_shift_requests[context]));
    if (do_send[1])
      MPI_Isend(
        src_cf_g->pool.data(),
        src_cf_g->size,
        mpi_datatype<impl::cf_t>::value(),
        dest,
        0,
        this->m_grid_comm,
        add_request(m_shift_requests[context]));
    if (do_recv[0])
      MPI_Irecv(
        dst_cf_d->pool.data(),
        dst_cf_d->size,
        mpi_datatype<impl::cf_t>::value(),
        source,
        0,
        this->m_grid_comm,
        add_request(m_shift_requests[context]));
    if (do_recv[1])
      MPI_Irecv(
        dst_cf_g->pool.data(),
        dst_cf_g->size,
        mpi_datatype<impl::cf_t>::value(),
        source,
        0,
        this->m_grid_comm,
        add_request(m_shift_requests[context]));
  }

  void
  shift_flags(
    unsigned context,
    ::hpg::runtime::StreamContext<D>& src_d,
    ::hpg::runtime::StreamContext<D>& dst_d,
    ::hpg::runtime::StreamContext<D>& src_g,
    ::hpg::runtime::StreamContext<D>& dst_g) const {

    int size_gc;
    MPI_Comm_size(this->m_grid_comm, &size_gc);
    int rank_gc;
    MPI_Comm_rank(this->m_grid_comm, &rank_gc);

    auto dest = (rank_gc + 1) % size_gc;
    auto source = (rank_gc + size_gc - 1) % size_gc;

    const MPI_Datatype dt = mpi_datatype<decltype(GriddingFlags::u)>::value();

    MPI_Irecv(
      &(std::any_cast<GriddingFlags>(&dst_d.m_user_data)->u),
      1,
      dt,
      source,
      ((rank_gc == 0) ? 1 : 0),
      this->m_grid_comm,
      add_request(m_shift_requests[context]));

    MPI_Irecv(
      &(std::any_cast<GriddingFlags>(&dst_g.m_user_data)->u),
      1,
      dt,
      source,
      ((rank_gc == 0) ? 0 : 1),
      this->m_grid_comm,
      add_request(m_shift_requests[context]));

    MPI_Isend(
      &(std::any_cast<GriddingFlags>(&src_d.m_user_data)->u),
      1,
      dt,
      dest,
      0,
      this->m_grid_comm,
      add_request(m_shift_requests[context]));

    MPI_Isend(
      &(std::any_cast<GriddingFlags>(&src_g.m_user_data)->u),
      1,
      dt,
      dest,
      1,
      this->m_grid_comm,
      add_request(m_shift_requests[context]));
  }

public:

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  set_convolution_function(
    unsigned context,
    Device host_device,
    CFArray&& cf_array) override {

    // N.B: access cf_array directly only at the root rank of m_grid_comm

    if (this->is_grid_partition_root()) {
      using DevCFArray = typename impl::DeviceCFArray<D>;
      DevCFArray dev_cf_array;
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

      // copy the CF kernels to device memory
      auto& ctx =
        this->m_exec_contexts[this->degrid_execution_context(context)];
      ctx.switch_to_copy(true);
      auto& cf = ctx.current_cf_pool();
      cf->add_device_cfs(ctx.current_exec_space(), std::move(dev_cf_array));
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
    bool do_grid,
    bool fence) const {

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
      m_shift_requests[context].size(),
      m_shift_requests[context].data(),
      MPI_STATUSES_IGNORE);
    m_shift_requests[context].clear();

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

      auto flags = std::any_cast<GriddingFlags>(&sc.m_user_data);
      m_pipeline_flush[context] = m_pipeline_flush[context] || fence;
      m_pipeline_flush[context] = m_pipeline_flush[context] && !flags->b.fence;
      flags->b.update_grid_weights = update_grid_weights;
      flags->b.do_degrid = do_degrid;
      flags->b.return_visibilities = return_visibilities;
      flags->b.do_grid = do_grid;
      flags->b.fence = fence;

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
      auto flags = std::any_cast<GriddingFlags>(sc.m_user_data);

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
      if (flags.b.do_degrid)
        gridder.degrid_all();

      // compute predicted or residual visibilities at tail rank
      //
      if (rank_gc == size_gc - 1) {
        if (flags.b.do_degrid)
          if (flags.b.do_grid)
            gridder.vis_copy_residual_and_rescale();
          else
            gridder.vis_copy_predicted();
        else
          gridder.vis_rescale();
      }
    }

    // do gridding (in gridding context)
    //
    {
      auto& ctx =
        this->m_exec_contexts[this->grid_execution_context(context)];
      ctx.switch_to_compute();
      auto& sc = ctx.current_stream_context();
      auto& cf = ctx.current_cf_pool();
      auto flags = std::any_cast<GriddingFlags>(sc.m_user_data);
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

      if (flags.b.do_grid) {
        if (flags.b.update_grid_weights)
          gridder.grid_all();
        else
          gridder.grid_all_no_weights();

        this->m_reduced_grid = this->m_reduced_grid && (num_vis == 0);
        this->m_reduced_weights =
          this->m_reduced_weights
          && (num_vis == 0 || !flags.b.update_grid_weights);
      }
      // return visibilities at root rank in gridding execution context
      //
      if (rank_gc == 0)
        result = sc.copy_visibilities_to_host(flags.b.return_visibilities);
    }

    {
      // do pipeline shift of flags and data (visibilities, weights and maybe
      // CFs) in both contexts
      //
      auto& ctx_d =
        this->m_exec_contexts[this->degrid_execution_context(context)];
      ctx_d.switch_to_compute();
      auto& src_d = ctx_d.current_stream_context();

      auto& ctx_g =
        this->m_exec_contexts[this->grid_execution_context(context)];
      ctx_g.switch_to_compute();
      auto& src_g = ctx_g.current_stream_context();

      shift_cfs(context, ctx_d, ctx_g);

      auto& dst_d = ctx_d.current_stream_context();
      auto& dst_g = ctx_g.current_stream_context();

      shift_context_data(
        context,
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.template visibilities<N>(); });
      shift_context_data(
        context,
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.m_weight_values; });
      shift_context_data(
        context,
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.m_weight_col_index; });
      shift_context_data(
        context,
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.m_weight_row_index; });
      src_d.fence(); // degridding must complete before shifting gvis
      shift_context_data(
        context,
        src_d,
        dst_d,
        src_g,
        dst_g,
        [](auto& sc) { return sc.template gvis<N>(); });

      shift_flags(context, src_d, dst_d, src_g, dst_g);
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
          do_grid,
          false);
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

protected:

  virtual void
  fence_unlocked() const noexcept override {
    bool fence_sentinel = true;
    bool flushing;
    do {
      for (unsigned c = 0; c < this->m_exec_contexts.size(); ++c)
        default_grid_visibilities<1>(
            c,
            D,
            std::vector<::hpg::VisData<1>>(),
            std::vector<vis_weight_fp>(),
            std::vector<unsigned>(),
            std::vector<size_t>(),
            false,
            false,
            false,
            false,
            fence_sentinel);
      fence_sentinel = false;
      // checking m_pipeline_flush in any single context is sufficient to decide
      // when the pipeline is in the flushing state
      assert(
        std::all_of(
          m_pipeline_flush.begin() + 1,
          m_pipeline_flush.end(),
          [f0 = m_pipeline_flush[0]](auto&& f) { return f == f0; }));
      flushing = m_pipeline_flush[0];
      MPI_Bcast(
        &flushing,
        1,
        mpi_datatype<decltype(flushing)>::value(),
        0,
        this->m_grid_comm);
    } while (flushing);
    this->m_exec_contexts.fence();
  }
};

} // end namespace hpg::mpi::runtime

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
