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

namespace hpg::mpi {

namespace K = Kokkos;

template <>
constexpr MPI_Datatype
mpi_datatype<K::complex<float>>() {
  return MPI_CXX_COMPLEX;
};
template <>
constexpr MPI_Datatype
mpi_datatype<K::complex<double>>() {
  return MPI_CXX_DOUBLE_COMPLEX;
};
template <>
constexpr MPI_Datatype
mpi_datatype<K::complex<long double>>() {
  return MPI_CXX_LONG_DOUBLE_COMPLEX;
};
} // end namespace hpg::mpi

namespace hpg::mpi::runtime {

namespace impl = ::hpg::runtime::impl;

namespace K = Kokkos;

template <unsigned N>
MPI_Datatype
visdata_datatype() {
  static std::mutex m;
  static MPI_Datatype result = MPI_DATATYPE_NULL;
  std::lock_guard<std::mutex> l(m);
  if (result == MPI_DATATYPE_NULL) {
    using VD = ::hpg::VisData<N>;
    constexpr int count = 8;
    int blocklengths[count] = {
      VD::npol, // m_visibilities
      VD::npol, // m_weights
      1, // m_frequency
      1, // m_phase
      std::declval<decltype(VD::m_uvw)>().size(), // m_uvw
      1, // m_grid_cube
      std::declval<decltype(VD::m_cf_index)>().size(), // m_cf_index
      std::declval<decltype(VD::m_cf_phase_gradient)>().size() // m_cf_phase_gradient
    };
    MPI_Aint displacements[count] = {
      offsetof(VD, m_visibilities),
      offsetof(VD, m_weights),
      offsetof(VD, m_frequency),
      offsetof(VD, m_phase),
      offsetof(VD, m_uvw),
      offsetof(VD, m_grid_cube),
      offsetof(VD, m_cf_index),
      offsetof(VD, m_cf_phase_gradient)
    };
    MPI_Datatype types[count] = {
      mpi_datatype<decltype(VD::m_visibilities)::value_type>(),
      mpi_datatype<decltype(VD::m_weights)>(),
      mpi_datatype<decltype(VD::m_frequency)>(),
      mpi_datatype<decltype(VD::m_phase)>(),
      mpi_datatype<decltype(VD::m_uvw)::value_type>(),
      mpi_datatype<decltype(VD::m_grid_cube)>(),
      mpi_datatype<decltype(VD::m_cf_index)::value_type>(),
      mpi_datatype<decltype(VD::m_cf_phase_gradient)::value_type>()
    };
    MPI_Type_create_struct(count, blocklengths, displacements, types, &result);
    MPI_Type_commit(&result);
  }
  return result;
}

template <unsigned N>
MPI_Datatype
gvisbuff_datatype() {
  static std::mutex m;
  static MPI_Datatype result = MPI_DATATYPE_NULL;
  std::lock_guard<std::mutex> l(m);
  if (result == MPI_DATATYPE_NULL) {
    using PA = ::hpg::runtime::impl::core::poln_array_type<visibility_fp, 4>;
    MPI_Datatype blk;
    MPI_Type_contiguous(
      N,
      mpi_datatype<std::complex<visibility_fp>>(),
      &blk);
    MPI_Type_create_resized(blk, 0, sizeof(PA), &result);
    MPI_Type_free(&blk);
    MPI_Type_commit(&result);
  }
  return result;
}

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

  // global offset of local grid cube indexes
  unsigned m_grid_cube_offset;

  mutable bool m_reduced_grid;

  mutable bool m_reduced_weights;

protected:

  State()
    : m_vis_comm(MPI_COMM_NULL)
    , m_grid_comm(MPI_COMM_NULL)
    , m_grid_cube_offset(0)
    , m_reduced_grid(true)
    , m_reduced_weights(true) {}

public:

  State(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    unsigned grid_cube_offset)
    : m_vis_comm(vis_comm)
    , m_grid_comm(grid_comm)
    , m_grid_cube_offset(grid_cube_offset)
    , m_reduced_grid(true)
    , m_reduced_weights(true) {}

  virtual ~State() {
    for (auto& c : {&m_vis_comm, &m_grid_comm}) {
      if (*c != MPI_COMM_NULL && *c != MPI_COMM_SELF && *c != MPI_COMM_WORLD)
        MPI_Comm_free(c);
    }
  }

  unsigned
  grid_cube_offset() const noexcept {
    return m_grid_cube_offset;
  }

  unsigned
  grid_cube_size() const noexcept {
    return grid_size()[unsigned(GridValueArray::Axis::cube)];
  }

  bool
  is_visibility_partition_root() const noexcept {
    int rank;
    MPI_Comm_rank(m_vis_comm, &rank);
    return rank == 0;
  }

  bool
  is_grid_partition_root() const noexcept {
    int rank;
    MPI_Comm_rank(m_grid_comm, &rank);
    return rank == 0;
  }
};

template <Device D>
struct /*HPG_EXPORT*/ StateT
  : public ::hpg::runtime::StateT<D>
  , public State {

  using typename ::hpg::runtime::StateT<D>::maybe_vis_t;

  using typename ::hpg::runtime::StateT<D>::kokkos_device;
  using typename ::hpg::runtime::StateT<D>::execution_space;
  using typename ::hpg::runtime::StateT<D>::memory_space;
  using typename ::hpg::runtime::StateT<D>::device_traits;
  using typename ::hpg::runtime::StateT<D>::stream_type;
  using typename ::hpg::runtime::StateT<D>::grid_layout;

  using SD = typename ::hpg::runtime::StateT<D>;

  using StreamPhase = ::hpg::runtime::StreamPhase;

public:

  StateT(
    MPI_Comm vis_comm,
    MPI_Comm grid_comm,
    unsigned grid_cube_offset,
    unsigned max_active_tasks,
    size_t max_visibility_batch_size,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4> grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const IArrayVector& mueller_indexes,
    const IArrayVector& conjugate_mueller_indexes,
    const std::array<unsigned, 4>& implementation_versions)
    : State(vis_comm, grid_comm, grid_cube_offset)
    , ::hpg::runtime::StateT<D>(
      max_active_tasks,
      max_visibility_batch_size,
      init_cf_shape,
      grid_size,
      grid_scale,
      mueller_indexes,
      conjugate_mueller_indexes,
      implementation_versions) {}

  StateT(const StateT& st) = delete;

  StateT(StateT&& st) noexcept
    : State()
    , ::hpg::runtime::StateT<D>(std::move(st)) {

    std::swap(m_reduced_grid, st.m_reduced_grid);
    std::swap(m_reduced_weights, st.m_reduced_weights);
    std::swap(m_vis_comm, st.m_vis_comm);
    std::swap(m_grid_comm, st.m_grid_comm);
  }

  virtual ~StateT() {}

  StateT&
  operator=(const StateT& st) = delete;

  StateT&
  operator=(StateT&& st) noexcept {
    StateT tmp(std::move(st));
    this->swap(tmp);
    return *this;
  }

  bool
  non_trivial_visibility_partition() const noexcept {
    int size;
    MPI_Comm_size(m_vis_comm, &size);
    return size > 1;
  }

  bool
  non_trivial_grid_partition() const noexcept {
    int size;
    MPI_Comm_size(m_grid_comm, &size);
    return size > 1;
  }

  void
  reduce_weights_unlocked() const {
    if (!m_reduced_weights) {
      if (non_trivial_visibility_partition()) {
        auto is_root = is_visibility_partition_root();
        auto& exec =
          this->m_exec_spaces[
            this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
        exec.fence();
        MPI_Reduce(
          (is_root ? MPI_IN_PLACE : this->m_weights.data()),
          this->m_weights.data(),
          this->m_weights.span(),
          mpi_datatype<grid_value_fp>(),
          MPI_SUM,
          0,
          m_vis_comm);
        if (!is_root)
          const_cast<StateT<D>*>(this)->fill_weights(grid_value_fp(0));
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
      if (non_trivial_visibility_partition()) {
        auto is_root = is_visibility_partition_root();
        auto& exec =
          this->m_exec_spaces[
            this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
        exec.fence();
        MPI_Reduce(
          (is_root ? MPI_IN_PLACE : this->m_grid.data()),
          this->m_grid.data(),
          this->m_grid.span(),
          mpi_datatype<impl::core::gv_t>(),
          MPI_SUM,
          0,
          m_vis_comm);
        if (!is_root)
          const_cast<StateT<D>*>(this)->fill_grid(impl::core::gv_t(0));
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

    // check that cf_array support isn't larger than grid
    {
      bool exceeds_grid = false;
      if (is_root) {
        for (unsigned grp = 0;
             !exceeds_grid && grp < cf_array.num_groups();
             ++grp) {
          auto extents = cf_array.extents(grp);
          if ((extents[CFArray::Axis::x] >
               this->m_grid_size[int(impl::core::GridAxis::x)]
               * cf_array.oversampling())
              || (extents[CFArray::Axis::y] >
                  this->m_grid_size[int(impl::core::GridAxis::y)]
                  * cf_array.oversampling()))
            exceeds_grid = true;
        }
      }
      MPI_Bcast(&exceeds_grid, 1, mpi_datatype<bool>(), 0, m_grid_comm);
      if (exceeds_grid)
        return std::make_unique<CFSupportExceedsGridError>();
    }

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
    MPI_Bcast(&shape_sz, 1, mpi_datatype<int>(), 0, m_grid_comm);
    shape.resize(shape_sz);
    MPI_Bcast(
      shape.data(),
      shape_sz,
      mpi_datatype<unsigned>(),
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
        mpi_datatype<CFArray::value_type>(),
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

    // N.B: access gv directly only at the root rank of m_vis_comm
    bool is_vis_root = is_visibility_partition_root();

    // check that gv size equals that of grid
    {
      std::array<unsigned, 4> model_sz;
      if (is_vis_root)
        model_sz = {gv.extent(0), gv.extent(1), gv.extent(2), gv.extent(3)};
      MPI_Bcast(
        model_sz.data(),
        4,
        mpi_datatype<unsigned>(),
        0,
        m_vis_comm);
      if (this->m_grid_size != model_sz)
        return
          std::make_unique<InvalidModelGridSizeError>(
            model_sz,
            this->m_grid_size);
    }

    fence();
    auto& exec =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::PRE_GRIDDING)];

    if (!this->m_model.is_allocated()) {
      std::array<int, 4> ig{
        int(this->m_grid_size[0]),
        int(this->m_grid_size[1]),
        int(this->m_grid_size[2]),
        int(this->m_grid_size[3])};
      this->m_model =
        decltype(this->m_model)(
          K::ViewAllocateWithoutInitializing("model"),
          SD::grid_layout::dimensions(ig));
    }

    // copy the model to device memory on the root rank, then broadcast it to
    // the other ranks
    if (is_vis_root) {
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
      if (non_trivial_visibility_partition())
        exec.fence();
    }
    // broadcast the model values
    MPI_Bcast(
      this->m_model.data(),
      this->m_model.span(),
      mpi_datatype<impl::core::gv_t>(),
      0,
      m_vis_comm);

    return std::nullopt;
  }

  template <unsigned N>
  maybe_vis_t
  default_grid_visibilities(
    Device /*host_device*/,
    std::vector<::hpg::VisData<N>>&& visibilities,
    bool update_grid_weights,
    bool do_degrid,
    bool return_visibilities,
    bool do_grid) {

    // Broadcast visibilities from grid partition subspace root rank. Do this
    // after copying visibilities to the device on the root rank to take
    // advantage potentially of high b/w device-device interconnects.
    auto is_grid_root = is_grid_partition_root();

    auto& exec_pre =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::PRE_GRIDDING)];
    MPI_Aint len;
    if (is_grid_root) {
      len = exec_pre.copy_visibilities_to_device(std::move(visibilities));
      if (non_trivial_grid_partition())
        exec_pre.fence();
    }
    MPI_Bcast(&len, 1, MPI_AINT, 0, m_grid_comm);
    exec_pre.num_visibilities = len;
    {
      auto vis = exec_pre.template visdata<N>();
      MPI_Bcast(
        vis.data(),
        len * sizeof(typename decltype(vis)::value_type),
        MPI_BYTE,
        0,
        m_grid_comm);
    }
    m_reduced_grid = m_reduced_grid && (len == 0);
    m_reduced_weights = m_reduced_grid && (len == 0 || !update_grid_weights);

    auto& exec_grid =
      this->m_exec_spaces[this->next_exec_space(StreamPhase::GRIDDING)];
    auto& cf = std::get<0>(this->m_cfs[this->m_cf_indexes.front()]);
    impl::core::const_grid_view<typename grid_layout::layout, memory_space>
      model = this->m_model;

    {
      // TODO: kernel version should be changed to 0
      using gridder =
        typename impl::core::VisibilityGridder<N, execution_space, 1>;
      auto espace = exec_grid.space;
      auto vis = exec_grid.template visdata<N>();
      auto& gvisbuff = exec_grid.gvisbuff;

      if (do_degrid) {
        gridder::degrid_all(
          espace,
          cf.cf_d,
          cf.cf_radii,
          cf.max_cf_extent_y,
          this->m_mueller_indexes,
          this->m_conjugate_mueller_indexes,
          len,
          vis,
          gvisbuff,
          this->m_grid_scale,
          model,
          this->m_grid,
          int(m_grid_cube_offset));
        if (non_trivial_grid_partition())
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
          4 * len,
          mpi_datatype<K::complex<visibility_fp>>(),
          MPI_SUM,
          m_grid_comm);
        if (do_grid)
          gridder::vis_copy_residual_and_rescale(espace, len, vis, gvisbuff);
        else
          gridder::vis_copy_predicted(espace, len, vis, gvisbuff);
      } else {
        gridder::vis_rescale(espace, len, vis, gvisbuff);
      }

      if (do_grid) {
        if (update_grid_weights)
          gridder::grid_all(
            espace,
            cf.cf_d,
            cf.cf_radii,
            cf.max_cf_extent_y,
            this->m_mueller_indexes,
            this->m_conjugate_mueller_indexes,
            len,
            vis,
            gvisbuff,
            this->m_grid_scale,
            this->m_grid,
            this->m_weights,
            int(m_grid_cube_offset));
        else
          gridder::grid_all_no_weights(
            espace,
            cf.cf_d,
            cf.cf_radii,
            cf.max_cf_extent_y,
            this->m_mueller_indexes,
            this->m_conjugate_mueller_indexes,
            len,
            vis,
            gvisbuff,
            this->m_grid_scale,
            this->m_grid,
            int(m_grid_cube_offset));
      }
    }
    // do not return visibilities to host if this is not the grid partition
    // subspace root rank
    return
      exec_grid.copy_visibilities_to_host(return_visibilities && is_grid_root);
  }

  template <unsigned N>
  maybe_vis_t
  grid_visibilities(
    Device host_device,
    std::vector<::hpg::VisData<N>>&& visibilities,
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
    if (is_visibility_partition_root()) {
      auto& exec =
        this->m_exec_spaces[
          this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
      auto wgts_h = K::create_mirror(this->m_weights);
      K::deep_copy(exec.space, wgts_h, this->m_weights);
      exec.fence();
      return std::make_unique<impl::GridWeightViewArray<D>>(wgts_h);
    } else {
      return nullptr;
    }
  }

  virtual std::unique_ptr<GridValueArray>
  grid_values() const override {
    std::scoped_lock lock(this->m_mtx);
    reduce_grid_unlocked();
    if (is_visibility_partition_root()) {
      auto& exec =
        this->m_exec_spaces[
          this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
      auto grid_h = K::create_mirror(this->m_grid);
      K::deep_copy(exec.space, grid_h, this->m_grid);
      exec.fence();
      return std::make_unique<impl::GridValueViewArray<D>>(grid_h);
    } else {
      return nullptr;
    }
  }

  virtual std::unique_ptr<GridValueArray>
  model_values() const override {
    std::scoped_lock lock(this->m_mtx);
    this->fence_unlocked();
    if (is_visibility_partition_root()) {
      if (this->m_model.is_allocated()) {
        auto& exec =
          this->m_exec_spaces[
            this->next_exec_space_unlocked(StreamPhase::PRE_GRIDDING)];
        auto model_h = K::create_mirror(this->m_model);
        K::deep_copy(exec.space, model_h, this->m_model);
        exec.fence();
        return std::make_unique<impl::GridValueViewArray<D>>(model_h);
      } else {
        std::array<unsigned, 4> ex{
          unsigned(this->m_grid.extent(0)),
          unsigned(this->m_grid.extent(1)),
          unsigned(this->m_grid.extent(2)),
          unsigned(this->m_grid.extent(3))};
        return std::make_unique<impl::UnallocatedModelValueArray>(ex);
      }
    } else {
      return nullptr;
    }
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
    if (is_visibility_partition_root()) {
      auto& exec =
        this->m_exec_spaces[
          this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
      impl::core::const_weight_view<
        typename execution_space::array_layout, memory_space>
        cweights = this->m_weights;
      switch (grid_normalizer_version()) {
      case 0:
        impl::core::GridNormalizer<execution_space, 0>::kernel(
          this->m_exec_spaces[
            this->next_exec_space(StreamPhase::GRIDDING)].space,
          this->m_grid,
          cweights,
          wfactor);
        break;
      default:
        assert(false);
        break;
      }
    }
  }

  virtual std::optional<std::unique_ptr<::hpg::Error>>
  apply_grid_fft(grid_value_fp norm, FFTSign sign, bool in_place) override {
    reduce_grid();
    std::optional<std::unique_ptr<::hpg::Error>> err;
    if (is_visibility_partition_root()) {
      auto& exec =
        this->m_exec_spaces[
          this->next_exec_space_unlocked(StreamPhase::GRIDDING)];
      if (in_place) {
        switch (fft_version()) {
        case 0:
          err =
            impl::core::FFT<execution_space, 0>
            ::in_place_kernel(
              this->m_exec_spaces[
                this->next_exec_space(StreamPhase::GRIDDING)].space,
              sign,
              this->m_grid);
          break;
        default:
          assert(false);
          break;
        }
      } else {
        impl::core::const_grid_view<typename grid_layout::layout, memory_space>
          pre_grid = this->m_grid;
        this->new_grid(false, false);
        switch (fft_version()) {
        case 0:
          err =
            impl::core::FFT<execution_space, 0>::out_of_place_kernel(
              this->m_exec_spaces[
                this->next_exec_space(StreamPhase::GRIDDING)].space,
              sign,
              pre_grid,
              this->m_grid);
          break;
        default:
          assert(false);
          break;
        }
      }
      // apply normalization
      if (norm != 1)
        switch (grid_normalizer_version()) {
        case 0:
          impl::core::GridNormalizer<execution_space, 0>::kernel(
            this->m_exec_spaces[
              this->next_exec_space(StreamPhase::GRIDDING)].space,
            this->m_grid,
            norm);
          break;
        default:
          assert(false);
          break;
        }
    }
    return err;
  }

protected:

  void
  swap(StateT& other) noexcept {
    ::hpg::runtime::StateT<D>::swap(
      static_cast<::hpg::runtime::StateT<D>&>(other));
    std::swap(m_vis_comm, other.m_vis_comm);
    std::swap(m_grid_comm, other.m_grid_comm);
    std::swap(m_grid_cube_offset, other.m_grid_cube_offset);
    std::swap(m_reduced_grid, other.m_reduced_grid);
    std::swap(m_reduced_weights, other.m_reduced_weights);
  }
};

} // end namespace hpg::mpi::runtime

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
