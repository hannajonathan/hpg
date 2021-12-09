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

#include "hpg.hpp"

#ifdef HPG_ENABLE_MPI

#include <mpi.h>
#include <mutex>

namespace hpg::mpi {

/** MPI error types
 */
enum class HPG_EXPORT ErrorType {
  OddNumberStreams,
  InvalidCommunicatorSize,
  NullCommunicator,
  NonconformingGridPartition,
  Other
};

/** error class
 */
class HPG_EXPORT Error
  : public ::hpg::Error {
private:

  ErrorType m_mpi_type;

public:

  /** error constructor */
  Error(const std::string& msg, ErrorType err = ErrorType::Other);

  Error() {}

  /** error type */
  ErrorType
    mpi_type() const;

  /** destructor */
  virtual ~Error();
};

struct OddNumberStreamsError
  : public Error {

  OddNumberStreamsError();
};

struct InvalidCommunicatorSizeError
  : public Error {

  InvalidCommunicatorSizeError();
};

struct NullCommunicatorError
  : public Error {

  NullCommunicatorError();
};

struct NonconformingGridPartitionError
  : public Error {

  NonconformingGridPartitionError();
};

struct ReplicatedGridDecomposition {
  // three axes are x, y, and channel; mrow partition not supported
  static constexpr unsigned rank = 3;
  enum Axis {x, y, channel};

  std::array<std::vector<unsigned>, rank> m_sizes;

  std::vector<std::vector<std::vector<unsigned>>> m_num_extra_replicas;

  ReplicatedGridDecomposition(
    const std::array<std::vector<unsigned>, rank>& sizes);

  bool
  conforms_to(
    const std::array<unsigned, GridValueArray::rank>& grid_size) const;

  unsigned
  size() const;
};

enum class VisibilityDistribution {
  Broadcast,
  Pipeline,
};

namespace GridderState {

  /** factory method
   *
   * @param device gridder device type
   * @param max_added_tasks maximum number of additional tasks; must be an odd
   * number (actual number may be less than requested)
   * @param visibility_batch_size batch size for number of VisData elements
   * (N.B: this value is currently a hard limit of the implementation that
   * governs the maximum number of elements in a vector of VisData elements
   * accepted by the grid_visibilities() family of methods)
   * @param max_avg_channels_per_vis maximum average (over
   * visibilities in a batch) number of channels onto which visibilities
   * are mapped
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size in logical axis order: X, Y, mrow, channel
   * @param grid_scale in X, Y order
   * @param mueller_indexes CFArray Mueller element indexes, by mrow
   * @param conjugate_mueller_indexes CFArray conjugate Mueller element indexes,
   * by mrow
   * @param comm MPI communicator with 2d topology, MPI_COMM_SELF, or
   * MPI_COMM_NULL
   * @param vis_part_size size of visibility partition (i.e, number of subsets
   * in partition)
   * @param grid_part grid partition
   * @param visibility_distribution visibility distribution algorithm for grid
   * partition
   *
   * max_added_tasks may be used to control the level of concurrency available
   * to the GridderState instance. In all cases, at least one task is
   * employed, but some devices support additional, concurrent tasks.
   *
   * The values of max_added_tasks, visibility_batch_size and
   * max_avg_channels_per_vis have effects on the amount of memory allocated
   * on the selected gridder device.
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @return tuple of (GridderState, true iff root rank of visibility partition,
   * true iff root rank of grid partition)
   */
  rval_t<std::tuple<::hpg::GridderState, bool, bool>>
  create(
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
    VisibilityDistribution visibility_distribution) noexcept;

  /** factory method
   *
   * @param device gridder device type
   * @param max_added_tasks maximum number of additional tasks; must be an odd
   * number (actual number may be less than requested)
   * @param visibility_batch_size batch size for number of VisData elements
   * (N.B: this value is currently a hard limit of the implementation that
   * governs the maximum number of elements in a vector of VisData elements
   * accepted by the grid_visibilities() family of methods)
   * @param max_avg_channels_per_vis maximum average (over
   * visibilities in a batch) number of channels onto which visibilities
   * are mapped
   * @param init_cf_shape shape of CF region for initial memory allocation (per
   * task)
   * @param grid_size in logical axis order: X, Y, mrow, channel
   * @param grid_scale in X, Y order
   * @param mueller_indexes CFArray Mueller element indexes, by mrow
   * @param conjugate_mueller_indexes CFArray conjugate Mueller element indexes,
   * by mrow
   * @param comm MPI communicator
   * @param vis_part_size size of visibility partition (i.e, number of subsets
   * in partition)
   * @param grid_part_size size of grid partition (i.e, number of subsets in
   * partition)
   * @param visibility_distribution visibility distribution algorithm for grid
   * partition
   *
   * Does not throw an exception if device argument names an unsupported
   * device. This function creates a new communicator from comm with a 2d
   * topology of the requested dimensions.
   *
   * @return tuple of (GridderState, true iff root rank of visibility partition,
   * true iff root rank of grid partition)
   */
  rval_t<std::tuple<::hpg::GridderState, bool, bool>>
  create2d(
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
    VisibilityDistribution visibility_distribution) noexcept;

  /** factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @tparam N number of polarizations in visibilities to be gridded
   *
   * @return tuple of (GridderState, true iff root rank of visibility partition,
   * true iff root rank of grid partition)
   */
  template <unsigned N>
  rval_t<std::tuple<::hpg::GridderState, bool, bool>>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    MPI_Comm comm,
    unsigned vis_part_size,
    const ReplicatedGridDecomposition& grid_part,
    VisibilityDistribution visibility_distribution) noexcept {

    return
      create(
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        comm,
        vis_part_size,
        grid_part,
        visibility_distribution);
  }

  /** factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @tparam N number of polarizations in visibilities to be gridded
   *
   * @return tuple of (GridderState, true iff root rank of visibility partition,
   * true iff root rank of grid partition)
   */
  template <unsigned N>
  rval_t<std::tuple<::hpg::GridderState, bool, bool>>
  create2d(
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    MPI_Comm comm,
    unsigned vis_part_size,
    unsigned grid_part_size,
    VisibilityDistribution visibility_distribution) noexcept {

    return
      create2d(
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        comm,
        vis_part_size,
        grid_part_size,
        visibility_distribution);
  }
} // end namespace GridderState

namespace Gridder {

  /** Gridder factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   */
  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create(
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
    VisibilityDistribution visibility_distribution) noexcept;

  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create2d(
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
    VisibilityDistribution visibility_distribution) noexcept;

  /** Gridder factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   *
   * @tparam N number of polarization in visibilities to be gridded
   *
   * @sa Gridder()
   */
  template <unsigned N>
  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create(
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    MPI_Comm comm,
    unsigned vis_part_size,
    const ReplicatedGridDecomposition& grid_part,
    VisibilityDistribution visibility_distribution) noexcept {

    return
      create(
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        comm,
        vis_part_size,
        grid_part,
        visibility_distribution);
  }

  template <unsigned N>
  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create2d(
    Device device,
    unsigned max_added_tasks,
    size_t visibility_batch_size,
    unsigned max_avg_channels_per_vis,
    const CFArrayShape* init_cf_shape,
    const std::array<unsigned, 4>& grid_size,
    const std::array<grid_scale_fp, 2>& grid_scale,
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes,
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    const std::array<unsigned, 4>& implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    MPI_Comm comm,
    unsigned vis_part_size,
    unsigned grid_part_size,
    VisibilityDistribution visibility_distribution) noexcept {

    return
      create2d(
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes),
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        implementation_versions,
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        comm,
        vis_part_size,
        grid_part_size,
        visibility_distribution);
  }

} // end namespace Gridder

} // end namespace hpg::mpi

#endif // HPG_ENABLE_MPI

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
