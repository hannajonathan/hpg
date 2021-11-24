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

InvalidCommunicatorSizeError::InvalidCommunicatorSizeError()
  : Error(
    "Communicator size too small",
    ErrorType::InvalidCommunicatorSize) {}

NullCommunicatorError::NullCommunicatorError()
  : Error(
    "Null communicator is invalid",
    ErrorType::NullCommunicator) {}

bool
ReplicatedGridBrick::disjoint(const std::vector<ReplicatedGridBrick>& bricks) {
  bool result = true;
  for (size_t i = 0; result && i < bricks.size(); ++i)
    result =
      std::all_of(
        bricks.begin() + i + 1,
        bricks.end(),
        [bi = bricks[i]](auto& b) {
          bool nonoverlapping = bi.num_replicas == 0 || b.num_replicas == 0;
          for (size_t j = 0; !nonoverlapping && j < b.offset.size(); ++j)
            nonoverlapping =
              bi.offset[j] + bi.size[j] <= b.offset[j]
              || b.offset[j] + b.size[j] <= bi.offset[j];
          return nonoverlapping;
        });
  return result;
}

// FIXME
//
// bool
// ReplicatedGridBrick::complete(
//   const std::vector<ReplicatedGridBrick>& bricks,
//   const std::array<unsigned, GridValueArray::rank>& space) {

//   std::array<unsigned, ReplicatedGridBrick::rank> min;
//   std::fill_n(min.begin(), min.size(), std::numeric_limits<unsigned>::max());
//   std::array<unsigned, ReplicatedGridBrick::rank> max;
//   std::fill_n(max.begin(), max.size(), std::numeric_limits<unsigned>::min());
//   for (auto& b : bricks)
//     if (b.num_replicas > 0)
//       for (size_t i = 0; i < min.size(); ++i) {
//         min[i] = std::min(min[i], b.offset[i]);
//         max[i] = std::max(max[i], b.offset[i] + b.size[i]);
//       }
//   return
//     std::all_of(min.begin(), min.end(), [](auto c) { return c == 0; })
//     && max[ReplicatedGridBrick::x] == space[GridValueArray::x]
//     && max[ReplicatedGridBrick::y] == space[GridValueArray::y]
//     && max[ReplicatedGridBrick::channel] == space[GridValueArray::channel];
// }

static std::shared_ptr<::hpg::mpi::runtime::State>
create_impl(
  ::hpg::Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const ::hpg::CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<::hpg::grid_scale_fp, 2>& grid_scale,
  const ::hpg::IArrayVector& mueller_indexes,
  const ::hpg::IArrayVector& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  MPI_Comm comm,
  unsigned vis_part_size,
  const std::vector<ReplicatedGridBrick>& grid_part) {

  using namespace ::hpg::mpi;

  // compute communicators for subspaces of visibility, grid and replica
  // partitions for this rank
  //
  // For example, assume no replicas > 1, 2d indexes are ordered (visibility
  // subset, grid plane subset), and we represent the processes in the group by
  //
  // r00 r01 r02
  // r10 r11 r12
  //
  // if this process is r01, its grid plane subspace is (r00 r01 r02), and its
  // visibility subspace is (r01 r11)
  //

  auto grid_part_size =
    std::accumulate(
      grid_part.begin(),
      grid_part.end(),
      0u,
      [](const unsigned& acc, const auto& b) { return acc + b.num_replicas; });

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm grid_comm; // grid partition comm
  MPI_Comm_split(
    comm,
    comm_rank / grid_part_size,
    comm_rank % grid_part_size,
    &grid_comm);
  MPI_Comm vis_comm; // visibility partition comm
  MPI_Comm_split(
    comm,
    comm_rank % grid_part_size,
    comm_rank / grid_part_size,
    &vis_comm);
  int grid_rank;
  MPI_Comm_rank(grid_comm, &grid_rank);
  ssize_t grid_brick_index = -1;
  MPI_Comm replica_comm = MPI_COMM_SELF;
  int replica_rank = 0;
  for (size_t i = 0; i < grid_part.size(); ++i) {
    bool my_brick =
      replica_rank <= grid_rank
      && grid_rank < replica_rank + grid_part[i].num_replicas;
    if (my_brick)
      grid_brick_index = i;
    if (grid_part[i].num_replicas > 1) {
      MPI_Comm c;
      MPI_Comm_split(grid_comm, my_brick ? 0 : MPI_UNDEFINED, 0, &c);
      if (my_brick)
        replica_comm = c;
    }
    replica_rank += grid_part[i].num_replicas;
  }

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
        grid_part[grid_brick_index],
        replica_comm,
        max_active_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
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
        grid_part[grid_brick_index],
        replica_comm,
        max_active_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
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
        grid_part[grid_brick_index],
        replica_comm,
        max_active_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
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
  Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  MPI_Comm comm,
  unsigned vis_part_size,
  const std::vector<ReplicatedGridBrick>& grid_part) noexcept {

  using val_t = std::tuple<::hpg::GridderState, bool, bool>;

  if (grid_size[2] != mueller_indexes.size()
      || grid_size[2] != conjugate_mueller_indexes.size())
    return rval<val_t>(std::make_unique<InvalidNumberMuellerIndexRowsError>());

  if (comm == MPI_COMM_NULL)
    return rval<val_t>(std::make_unique<NullCommunicatorError>());

  if (devices().count(device) == 0)
    return rval<val_t>(std::make_unique<DisabledDeviceError>());

  auto num_repl_grid_bricks =
    std::accumulate(
      grid_part.begin(),
      grid_part.end(),
      0u,
      [](const unsigned& acc, const auto& b) { return acc + b.num_replicas; });
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  if (comm_size < vis_part_size * std::max(num_repl_grid_bricks, 1u))
    return rval<val_t>(std::make_unique<InvalidCommunicatorSizeError>());
  std::vector<ReplicatedGridBrick> nonempty_grid_part;
  // ensure that the default nonempty_grid_part is constructed below using the
  // correct indexes
  static_assert(
    ReplicatedGridBrick::Axis::x == 0
    && ReplicatedGridBrick::Axis::y == 1
    && ReplicatedGridBrick::Axis::channel == 2);
  if (num_repl_grid_bricks == 0)
    nonempty_grid_part.push_back(
      ReplicatedGridBrick{
        1,
        {0, 0, 0},
        {grid_size[GridValueArray::Axis::x],
         grid_size[GridValueArray::Axis::y],
         grid_size[GridValueArray::Axis::channel]}});
  else
    std::copy_if(
      grid_part.begin(),
      grid_part.end(),
      std::back_inserter(nonempty_grid_part),
      [](const auto& b) { return b.num_replicas > 0; });

  auto impl =
    create_impl(
      device,
      max_added_tasks,
      visibility_batch_size,
      max_avg_channels_per_vis,
      init_cf_shape,
      grid_size,
      grid_scale,
      mueller_indexes,
      conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      implementation_versions,
#endif
      comm,
      vis_part_size,
      nonempty_grid_part);
  auto vis_part_root =
    impl->is_visibility_partition_root()
    && impl->is_replica_partition_root();
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
  Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  MPI_Comm comm,
  unsigned vis_part_size,
  unsigned grid_part_size) noexcept {

  using val_t = std::tuple<::hpg::GridderState, bool, bool>;

  // the grid plane partition divides the grid into segments on the channel
  // axis, we compute the offset and size of our local segment of channel
  // indexes
  //
  // Note that the following will compute offsets greater than the size of the
  // channel axis in some cases, which should be taken to indicate an empty grid
  // partition (computed segment size will be zero in this case, as well).
  //
  auto grid_channel_size = grid_size[::hpg::GridValueArray::Axis::channel];
  unsigned min_brick_channel_size = grid_channel_size / grid_part_size;
  unsigned brick_channel_rem_size = grid_channel_size % grid_part_size;
  std::vector<ReplicatedGridBrick> grid_bricks;
  for (unsigned i = 0; i < grid_part_size; ++i) {
    ReplicatedGridBrick brick;
    brick.num_replicas = 1;
    brick.offset[ReplicatedGridBrick::Axis::x] = 0;
    brick.offset[ReplicatedGridBrick::Axis::y] = 0;
    brick.offset[ReplicatedGridBrick::Axis::channel] =
      i * min_brick_channel_size + std::min(i, brick_channel_rem_size);
    brick.size[ReplicatedGridBrick::Axis::x] =
      grid_size[::hpg::GridValueArray::Axis::x];
    brick.size[ReplicatedGridBrick::Axis::y] =
      grid_size[::hpg::GridValueArray::Axis::y];
    brick.size[ReplicatedGridBrick::Axis::channel] =
      min_brick_channel_size + ((i < brick_channel_rem_size) ? 1 : 0);
    grid_bricks.push_back(std::move(brick));
  }
  assert(disjoint(grid_bricks));
  //assert(complete(grid_bricks, grid_size));

  return
    create(
      device,
      max_added_tasks,
      visibility_batch_size,
      max_avg_channels_per_vis,
      init_cf_shape,
      grid_size,
      grid_scale,
      std::move(mueller_indexes),
      std::move(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      comm,
      vis_part_size,
      grid_bricks);
}

static ::hpg::rval_t<std::tuple<::hpg::Gridder, bool, bool>>
apply_gridder(
  ::hpg::rval_t<std::tuple<::hpg::GridderState, bool, bool>>&& create_gs) {

  using rc_t = ::hpg::rval_t<std::tuple<::hpg::Gridder, bool, bool>>;

  return
    fold(
      std::move(create_gs),
      [](auto&& val) -> rc_t {
        auto [gs, vis_part_root, grid_part_root] = std::move(val);
        return
          std::make_tuple(
            ::hpg::Gridder(std::move(gs)),
            vis_part_root,
            grid_part_root);
      },
      [](auto&& err) -> rc_t {
        return std::move(err);
      });
}

::hpg::rval_t<std::tuple<::hpg::Gridder, bool, bool>>
Gridder::create(
  Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  MPI_Comm comm,
  unsigned vis_part_size,
  const std::vector<ReplicatedGridBrick>& grid_part) noexcept {

  return
    apply_gridder(
      GridderState::create(
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        std::move(mueller_indexes),
        std::move(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        comm,
        vis_part_size,
        grid_part));
}

::hpg::rval_t<std::tuple<::hpg::Gridder, bool, bool>>
Gridder::create2d(
  Device device,
  unsigned max_added_tasks,
  size_t visibility_batch_size,
  unsigned max_avg_channels_per_vis,
  const CFArrayShape* init_cf_shape,
  const std::array<unsigned, 4>& grid_size,
  const std::array<grid_scale_fp, 2>& grid_scale,
  IArrayVector&& mueller_indexes,
  IArrayVector&& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  MPI_Comm comm,
  unsigned vis_part_size,
  unsigned grid_part_size) noexcept {

  return
    apply_gridder(
      GridderState::create2d(
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        std::move(mueller_indexes),
        std::move(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        comm,
        vis_part_size,
        grid_part_size));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
