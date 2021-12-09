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

OddNumberStreamsError::OddNumberStreamsError()
  : Error(
    "Number of additional tasks must be odd",
    ErrorType::OddNumberStreams) {}

InvalidCommunicatorSizeError::InvalidCommunicatorSizeError()
  : Error(
    "Communicator size too small",
    ErrorType::InvalidCommunicatorSize) {}

NullCommunicatorError::NullCommunicatorError()
  : Error(
    "Null communicator is invalid",
    ErrorType::NullCommunicator) {}

NonconformingGridPartitionError::NonconformingGridPartitionError()
  : Error(
    "Grid partition does not conform to grid size",
    ErrorType::NonconformingGridPartition) {}

ReplicatedGridDecomposition::ReplicatedGridDecomposition(
  const std::array<std::vector<unsigned>, rank>& sizes)
  : m_sizes(sizes) {

  m_num_extra_replicas.resize(m_sizes[0].size());
  for (auto& r1 : m_num_extra_replicas) {
    r1.resize(m_sizes[1].size());
    for (auto& r2 : r1) {
      r2.resize(m_sizes[2].size());
      std::fill(r2.begin(), r2.end(), 0u);
    }
  }
}

bool
ReplicatedGridDecomposition::conforms_to(
  const std::array<unsigned, GridValueArray::rank>& grid_size) const {

  return
    (std::accumulate(m_sizes[Axis::x].begin(), m_sizes[Axis::x].end(), 0u)
     == grid_size[GridValueArray::Axis::x])
    && (std::accumulate(m_sizes[Axis::y].begin(), m_sizes[Axis::y].end(), 0u)
        == grid_size[GridValueArray::Axis::y])
    && (std::accumulate(
          m_sizes[Axis::channel].begin(),
          m_sizes[Axis::channel].end(),
          0u)
        == grid_size[GridValueArray::Axis::channel]);
}

unsigned
ReplicatedGridDecomposition::size() const {
  return
    std::accumulate(
      m_num_extra_replicas.begin(),
      m_num_extra_replicas.end(),
      0u,
      [](unsigned acc0, const auto& r0) {
        return
          std::accumulate(
            r0.begin(),
            r0.end(),
            acc0,
            [](unsigned acc1, const auto& r1) {
              return
                std::accumulate(
                  r1.begin(),
                  r1.end(),
                  acc1 + unsigned(r1.size()));
            });
      });
}

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
  const ReplicatedGridDecomposition& grid_part,
  VisibilityDistribution visibility_distribution) {

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

  auto grid_part_size = grid_part.size();

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  MPI_Comm grid_comm; // grid partition comm
  MPI_Comm_split(
    comm,
    comm_rank / grid_part_size,
    comm_rank % grid_part_size,
    &grid_comm);
  int grid_rank;
  MPI_Comm_rank(grid_comm, &grid_rank);
  MPI_Comm vis_comm; // visibility partition comm
  MPI_Comm_split(
    comm,
    comm_rank % grid_part_size,
    comm_rank / grid_part_size,
    &vis_comm);
  int vis_rank;
  MPI_Comm_rank(vis_comm, &vis_rank);

  std::vector<runtime::ReplicatedGridBrick> grid_bricks;
  grid_bricks.reserve(grid_part_size);
  {
    auto& sizes = grid_part.m_sizes;
    unsigned x0 = 0;
    for (size_t x = 0;
         x < sizes[ReplicatedGridDecomposition::Axis::x].size();
         ++x) {
      unsigned xsz = sizes[ReplicatedGridDecomposition::Axis::x][x];
      unsigned y0 = 0;
      for (size_t y = 0;
           y < sizes[ReplicatedGridDecomposition::Axis::y].size();
           ++y) {
        unsigned ysz = sizes[ReplicatedGridDecomposition::Axis::y][y];
        unsigned ch0 = 0;
        for (size_t ch = 0;
             ch < sizes[ReplicatedGridDecomposition::Axis::channel].size();
             ++ch) {
          unsigned chsz = sizes[ReplicatedGridDecomposition::Axis::channel][ch];
          grid_bricks.push_back(
            runtime::ReplicatedGridBrick{
              grid_part.m_num_extra_replicas[x][y][ch] + 1,
              {x0, y0, ch0},
              {xsz, ysz, chsz}});
          ch0 += chsz;
        }
        y0 += ysz;
      }
      x0 += xsz;
    }
  }
  int grid_brick_index = -1;
  int replica_rank = 0;
  bool has_replicas = false;
  int prev_ch = -1;
  int ch0 = -1;
  bool has_split_planes = false;
  for (size_t i = 0; i < grid_bricks.size(); ++i) {
    bool my_brick =
      replica_rank <= grid_rank
      && grid_rank < replica_rank + grid_bricks[i].num_replicas;
    int ch =
      int(grid_bricks[i].offset[runtime::ReplicatedGridBrick::Axis::channel]);
    if (my_brick) {
      grid_brick_index = i;
      ch0 = ch;
    }
    if (ch == prev_ch)
      has_split_planes = true;
    if (grid_bricks[i].num_replicas > 1)
      has_replicas = true;
    replica_rank += grid_bricks[i].num_replicas;
  }
  MPI_Comm replica_comm = MPI_COMM_SELF;
  if (has_replicas) {
    MPI_Comm c;
    MPI_Comm_split(
      grid_comm,
      ((grid_bricks[grid_brick_index].num_replicas > 1)
       ? grid_brick_index
       : MPI_UNDEFINED),
      0,
      &c);
    if (c != MPI_COMM_NULL)
      replica_comm = c;
  }
  MPI_Comm plane_comm = MPI_COMM_NULL;
  if (has_split_planes)
    MPI_Comm_split(
      grid_comm,
      ((vis_rank == 0) ? ch0 : MPI_UNDEFINED),
      0,
      &plane_comm);

  using namespace runtime;

#ifndef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
  std::array<unsigned, 4> implementation_versions{0, 0, 0, 0};
#endif

  const unsigned max_active_tasks = max_added_tasks + 1;

  std::shared_ptr<State> result;
  switch (device) {
#ifdef HPG_ENABLE_SERIAL
  case ::hpg::Device::Serial:
    switch (visibility_distribution) {
    case VisibilityDistribution::Broadcast:
      result =
        std::make_shared<
          StateT<::hpg::Device::Serial, VisibilityDistribution::Broadcast>>(
          vis_comm,
          grid_comm,
          grid_bricks[grid_brick_index],
          replica_comm,
          plane_comm,
          max_active_tasks,
          visibility_batch_size,
          max_avg_channels_per_vis,
          init_cf_shape,
          grid_size,
          grid_scale,
          mueller_indexes,
          conjugate_mueller_indexes,
          implementation_versions);
      break;
    case VisibilityDistribution::Pipeline:
      result =
        std::make_shared<
          StateT<::hpg::Device::Serial, VisibilityDistribution::Pipeline>>(
          vis_comm,
          grid_comm,
          grid_bricks[grid_brick_index],
          replica_comm,
          plane_comm,
          max_active_tasks,
          visibility_batch_size,
          max_avg_channels_per_vis,
          init_cf_shape,
          grid_size,
          grid_scale,
          mueller_indexes,
          conjugate_mueller_indexes,
          implementation_versions);
      break;
    default:
      assert(false);
      break;
    }
#else
    assert(false);
#endif // HPG_ENABLE_SERIAL
    break;
#ifdef HPG_ENABLE_OPENMP
  case ::hpg::Device::OpenMP:
    switch (visibility_distribution) {
    case VisibilityDistribution::Broadcast:
      result =
        std::make_shared<
          StateT<::hpg::Device::OpenMP, VisibilityDistribution::Broadcast>>(
            vis_comm,
            grid_comm,
            grid_bricks[grid_brick_index],
            replica_comm,
            plane_comm,
            max_active_tasks,
            visibility_batch_size,
            max_avg_channels_per_vis,
            init_cf_shape,
            grid_size,
            grid_scale,
            mueller_indexes,
            conjugate_mueller_indexes,
            implementation_versions);
      break;
    case VisibilityDistribution::Pipeline:
      result =
        std::make_shared<
          StateT<::hpg::Device::OpenMP, VisibilityDistribution::Pipeline>>(
            vis_comm,
            grid_comm,
            grid_bricks[grid_brick_index],
            replica_comm,
            plane_comm,
            max_active_tasks,
            visibility_batch_size,
            max_avg_channels_per_vis,
            init_cf_shape,
            grid_size,
            grid_scale,
            mueller_indexes,
            conjugate_mueller_indexes,
            implementation_versions);
      break;
    default:
      assert(false);
      break;
    }
#else
    assert(false);
#endif // HPG_ENABLE_OPENMP
    break;
#ifdef HPG_ENABLE_CUDA
  case ::hpg::Device::Cuda:
    switch (visibility_distribution) {
    case VisibilityDistribution::Broadcast:
      result =
        std::make_shared<
          StateT<::hpg::Device::Cuda, VisibilityDistribution::Broadcast>>(
            vis_comm,
            grid_comm,
            grid_bricks[grid_brick_index],
            replica_comm,
            plane_comm,
            max_active_tasks,
            visibility_batch_size,
            max_avg_channels_per_vis,
            init_cf_shape,
            grid_size,
            grid_scale,
            mueller_indexes,
            conjugate_mueller_indexes,
            implementation_versions);
      break;
    case VisibilityDistribution::Pipeline:
      result =
        std::make_shared<
          StateT<::hpg::Device::Cuda, VisibilityDistribution::Pipeline>>(
          vis_comm,
          grid_comm,
          grid_bricks[grid_brick_index],
          replica_comm,
          plane_comm,
          max_active_tasks,
          visibility_batch_size,
          max_avg_channels_per_vis,
          init_cf_shape,
          grid_size,
          grid_scale,
          mueller_indexes,
          conjugate_mueller_indexes,
          implementation_versions);
      break;
    default:
      assert(false);
      break;
    }
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
  const ReplicatedGridDecomposition& grid_part,
  VisibilityDistribution visibility_distribution) noexcept {

  using val_t = std::tuple<::hpg::GridderState, bool, bool>;

  if (max_added_tasks % 2 == 0)
    return rval<val_t>(std::make_unique<OddNumberStreamsError>());

  if (grid_size[2] != mueller_indexes.size()
      || grid_size[2] != conjugate_mueller_indexes.size())
    return rval<val_t>(std::make_unique<InvalidNumberMuellerIndexRowsError>());

  if (comm == MPI_COMM_NULL)
    return rval<val_t>(std::make_unique<NullCommunicatorError>());

  if (devices().count(device) == 0)
    return rval<val_t>(std::make_unique<DisabledDeviceError>());

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  if (comm_size < grid_part.size())
    return rval<val_t>(std::make_unique<InvalidCommunicatorSizeError>());

  if (!grid_part.conforms_to(grid_size))
    return rval<val_t>(std::make_unique<NonconformingGridPartitionError>());

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
      grid_part,
      visibility_distribution);
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
  unsigned grid_part_size,
  VisibilityDistribution visibility_distribution) noexcept {

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
  std::array<std::vector<unsigned>, ReplicatedGridDecomposition::rank>
    decomp_sizes;
  decomp_sizes[ReplicatedGridDecomposition::Axis::x]
    .push_back(grid_size[::hpg::GridValueArray::Axis::x]);
  decomp_sizes[ReplicatedGridDecomposition::Axis::y]
    .push_back(grid_size[::hpg::GridValueArray::Axis::y]);
  for (unsigned i = 0; i < grid_part_size; ++i)
    decomp_sizes[ReplicatedGridDecomposition::Axis::channel]
      .push_back(
        min_brick_channel_size + ((i < brick_channel_rem_size) ? 1 : 0));

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
      ReplicatedGridDecomposition(decomp_sizes),
      visibility_distribution);
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
  const ReplicatedGridDecomposition& grid_part,
  VisibilityDistribution visibility_distribution) noexcept {

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
        grid_part,
        visibility_distribution));
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
  unsigned grid_part_size,
  VisibilityDistribution visibility_distribution) noexcept {

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
        grid_part_size,
        visibility_distribution));
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
