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
#include "hpg_mpi.hpp"
#include "hpg_mpi_runtime.hpp"

#include <vector>

using namespace hpg::mpi;

/** error constructor */
Error::Error(const std::string& msg, ErrorType err)
  : ::hpg::Error(msg, ::hpg::ErrorType::Other)
  , m_mpi_type(err) {}

/** error type */
ErrorType
Error::mpi_type() const {
  return m_mpi_type;
}

Error::~Error() {}

InvalidTopologyError::InvalidTopologyError()
  : Error(
    "Communicator has unsupported topology (must be Cartesian or none)",
    ErrorType::InvalidTopology) {}

InvalidCartesianRankError::InvalidCartesianRankError()
  : Error(
    "Communicator Cartesian topology rank is unsupported (must be <= 2)",
    ErrorType::InvalidCartesianRank) {}

IdenticalPartitionIndexError::IdenticalPartitionIndexError()
  : Error(
    "Indexes of visibility and grid partitions in Cartesian topology "
    "are identical",
    ErrorType::IdenticalPartitionIndex) {}

InvalidPartitionIndexError::InvalidPartitionIndexError()
  : Error(
    "Index of visibility or grid partition is out of range",
    ErrorType::InvalidPartitionIndex) {}

NullCommunicatorError::NullCommunicatorError()
  : Error(
    "Null communicator is invalid",
    ErrorType::NullCommunicator) {}

InvalidPartitionSizeError::InvalidPartitionSizeError()
  : Error(
    "Size of visibility or grid partition is zero",
    ErrorType::InvalidPartitionSize) {}

InvalidGroupSizeError::InvalidGroupSizeError()
  : Error(
    "Size of communicator group is too small for desired partition",
    ErrorType::InvalidGroupSize) {}

static std::shared_ptr<::hpg::mpi::runtime::State>
create_impl(
  MPI_Comm comm,
  int vis_part_index,
  int grid_part_index,
  ::hpg::Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const ::hpg::CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<::hpg::grid_scale_fp, 2>& grid_scale,
  const ::hpg::IArrayVector& mueller_indexes,
  const ::hpg::IArrayVector& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  ) {

  using namespace ::hpg::mpi;

  // compute communicators for subspaces of visibility and grid plane partitions
  // for this rank
  //
  // For example, assume 2d indexes are ordered (visibility subset, grid plane
  // subset), and we represent the processes in the group by
  //
  // r00 r01 r02
  // r10 r11 r12
  //
  // if this process is r01, its grid plane subspace is (r00 r01 r02), and its
  // visibility subspace is (r01 r11)
  //
  MPI_Comm vis_comm;
  MPI_Comm grid_comm;
  int ndims = 1;
  int topo;
  MPI_Topo_test(comm, &topo);
  if (topo == MPI_CART)
    MPI_Cartdim_get(comm, &ndims);
  if (ndims == 1) {
    if (grid_part_index == -1) {
      MPI_Comm_dup(comm, &vis_comm);
      MPI_Comm_dup(MPI_COMM_SELF, &grid_comm);
    } else /* vis_part_index == -1 */ {
      MPI_Comm_dup(MPI_COMM_SELF, &vis_comm);
      MPI_Comm_dup(comm, &grid_comm);
    }
  } else {
    if (vis_part_index > -1) {
      std::vector<int> remain_dims(ndims);
      std::fill(remain_dims.begin(), remain_dims.end(), int(true));
      remain_dims[vis_part_index] = int(false);
      MPI_Cart_sub(comm, remain_dims.data(), &grid_comm);
    } else {
      MPI_Comm_dup(comm, &grid_comm);
    }
    if (grid_part_index > -1) {
      std::vector<int> remain_dims(ndims);
      std::fill(remain_dims.begin(), remain_dims.end(), int(true));
      remain_dims[grid_part_index] = int(false);
      MPI_Cart_sub(comm, remain_dims.data(), &vis_comm);
    } else {
      MPI_Comm_dup(comm, &vis_comm);
    }
  }

  // the grid plane partition divides the grid into segments on the channel
  // axis, we compute the offset and size of our local segment of channel
  // indexes
  //
  // Note that the following will compute offsets greater than the size of the
  // channel axis in some cases, which should be taken to indicate an empty grid
  // partition (computed segment size will be zero in this case, as well).
  //
  auto grid_channel_size =
    grid_size[unsigned(::hpg::GridValueArray::Axis::channel)];
  unsigned grid_channel_offset_local = 0;
  unsigned grid_channel_size_local = grid_channel_size;

  int grid_comm_size;
  MPI_Comm_size(grid_comm, &grid_comm_size);
  int grid_comm_rank;
  MPI_Comm_rank(grid_comm, &grid_comm_rank);
  unsigned min_grid_channel_size = grid_channel_size / unsigned(grid_comm_size);
  unsigned grid_channel_rem_size = grid_channel_size % unsigned(grid_comm_size);
  grid_channel_offset_local =
    unsigned(grid_comm_rank) * min_grid_channel_size
    + std::min(unsigned(grid_comm_rank), grid_channel_rem_size);
  grid_channel_size_local =
    min_grid_channel_size
    + ((unsigned(grid_comm_rank) < grid_channel_rem_size) ? 1 : 0);
  std::array<unsigned, 4> grid_size_local = grid_size;
  grid_size_local[unsigned(::hpg::GridValueArray::Axis::channel)] =
    grid_channel_size_local;

  using namespace runtime;

#ifndef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  std::array<unsigned, 4> implementation_versions{0, 0, 0, 0};
#endif

  const unsigned max_active_tasks = max_added_tasks + 1;

  std::shared_ptr<State> result;
  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case ::hpg::Device::Serial:
    result =
      std::make_shared<StateT<::hpg::Device::Serial>>(
        vis_comm,
        grid_comm,
        grid_channel_offset_local,
        max_active_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size_local,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
        implementation_versions);
#else
    assert(false);
#endif // HPG_ENABLE_SERIAL
    break;
#ifdef HPG_ENABLE_OPENMP
  case ::hpg::Device::OpenMP:
    result =
      std::make_shared<StateT<::hpg::Device::OpenMP>>(
        vis_comm,
        grid_comm,
        grid_channel_offset_local,
        max_active_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size_local,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
        implementation_versions);
#else
    assert(false);
#endif // HPG_ENABLE_OPENMP
    break;
#ifdef HPG_ENABLE_CUDA
  case ::hpg::Device::Cuda:
    result =
      std::make_shared<StateT<::hpg::Device::Cuda>>(
        vis_comm,
        grid_comm,
        grid_channel_offset_local,
        max_active_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size_local,
        grid_scale,
        mueller_indexes,
        conjugate_mueller_indexes,
        implementation_versions);
#else
    assert(false);
#endif //HPG_ENABLE_CUDA
    break;
  default:
    assert(false);
    break;
  }
  return result;
}

::hpg::rval_t<std::tuple<::hpg::GridderState, bool, bool>>
GridderState::create(
  MPI_Comm comm,
  int vis_part_index,
  int grid_part_index,
  Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  ) noexcept {

  using val_t = std::tuple<::hpg::GridderState, bool, bool>;

  if (grid_size[2] != mueller_indexes.size()
      || grid_size[2] != conjugate_mueller_indexes.size())
    return rval<val_t>(std::make_unique<InvalidNumberMuellerIndexRowsError>());

  if (comm == MPI_COMM_NULL)
    return rval<val_t>(std::make_unique<NullCommunicatorError>());

  int ndims = 1;
  int topo;
  MPI_Topo_test(comm, &topo);
  if (!(topo == MPI_UNDEFINED || topo == MPI_CART))
    return rval<val_t>(std::make_unique<InvalidTopologyError>());
  if (topo == MPI_CART) {
    MPI_Cartdim_get(comm, &ndims);
    if (ndims > 2)
      return rval<val_t>(std::make_unique<InvalidCartesianRankError>());
  }
  if (vis_part_index == grid_part_index)
    return rval<val_t>(std::make_unique<IdenticalPartitionIndexError>());
  if (vis_part_index >= ndims || grid_part_index >= ndims)
    return rval<val_t>(std::make_unique<InvalidPartitionIndexError>());

  if (devices().count(device) == 0)
    return rval<val_t>(std::make_unique<DisabledDeviceError>());

  auto impl =
    create_impl(
      comm,
      vis_part_index,
      grid_part_index,
      device,
      max_added_tasks,
      visibility_batch_size,
      max_avg_channels_per_vis,
      init_cf_shape,
      grid_size,
      grid_scale,
      mueller_indexes,
      conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      , implementation_versions
#endif
      );
  auto vis_part_root = impl->is_visibility_partition_root();
  auto grid_part_root = impl->is_grid_partition_root();
  return
    ::hpg::rval_t<val_t>(
      std::make_tuple(
        ::hpg::GridderState(impl),
        vis_part_root,
        grid_part_root));
}

::hpg::rval_t<std::tuple<::hpg::GridderState, bool, bool>>
GridderState::create2d(
  MPI_Comm comm,
  unsigned vis_part_size,
  unsigned grid_part_size,
  Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  , const std::array<unsigned, 4>& implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  ) noexcept {

  using val_t = std::tuple<::hpg::GridderState, bool, bool>;

  if (comm == MPI_COMM_NULL)
    return rval<val_t>(std::make_unique<NullCommunicatorError>());

  if (vis_part_size == 0 || grid_part_size == 0)
    return rval<val_t>(std::make_unique<InvalidPartitionSizeError>());

  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    if (unsigned(comm_size) < vis_part_size * grid_part_size)
      return rval<val_t>(std::make_unique<InvalidGroupSizeError>());
  }

  int dims[2]{int(vis_part_size), int(grid_part_size)};
  int periods[2]{0, 0};
  MPI_Comm comm_cart;
  MPI_Cart_create(comm, 2, dims, periods, true, &comm_cart);
  return
    create(
      comm_cart,
      0,
      1,
      device,
      max_added_tasks,
      visibility_batch_size,
      max_avg_channels_per_vis,
      init_cf_shape,
      grid_size,
      grid_scale,
      std::move(mueller_indexes),
      std::move(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      );
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
