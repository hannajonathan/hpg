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
  InvalidTopology,
  InvalidCartesianRank,
  IdenticalPartitionIndex,
  InvalidPartitionIndex,
  NullCommunicator,
  InvalidPartitionSize,
  InvalidGroupSize,
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


struct InvalidTopologyError
  : public Error {

  InvalidTopologyError();
};

struct InvalidCartesianRankError
  : public Error {

  InvalidCartesianRankError();
};

struct IdenticalPartitionIndexError
  : public Error {

  IdenticalPartitionIndexError();
};

struct InvalidPartitionIndexError
  : public Error {

  InvalidPartitionIndexError();
};

struct NullCommunicatorError
  : public Error {

  NullCommunicatorError();
};

struct InvalidPartitionSizeError
  : public Error {

  InvalidPartitionSizeError();
};

struct InvalidGroupSizeError
  : public Error {

  InvalidGroupSizeError();
};

template <typename T>
constexpr MPI_Datatype
mpi_datatype() {
  return MPI_DATATYPE_NULL;
}
template <>
constexpr MPI_Datatype
mpi_datatype<char>() {
  return MPI_CHAR;
}
template <>
constexpr MPI_Datatype
mpi_datatype<signed short int>() {
  return MPI_SHORT;
}
template <>
constexpr MPI_Datatype
mpi_datatype<signed int>() {
  return MPI_INT;
};
template <>
constexpr MPI_Datatype
mpi_datatype<signed long int>() {
  return MPI_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<signed long long int>() {
  return MPI_LONG_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<signed char>() {
  return MPI_SIGNED_CHAR;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned char>() {
  return MPI_UNSIGNED_CHAR;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned short int>() {
  return MPI_UNSIGNED_SHORT;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned int>() {
  return MPI_UNSIGNED;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned long int>() {
  return MPI_UNSIGNED_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<unsigned long long int>() {
  return MPI_UNSIGNED_LONG_LONG;
};
template <>
constexpr MPI_Datatype
mpi_datatype<float>() {
  return MPI_FLOAT;
};
template <>
constexpr MPI_Datatype
mpi_datatype<double>() {
  return MPI_DOUBLE;
};
template <>
constexpr MPI_Datatype
mpi_datatype<long double>() {
  return MPI_LONG_DOUBLE;
};
template <>
constexpr MPI_Datatype
mpi_datatype<wchar_t>() {
  return MPI_WCHAR;
};
template <>
constexpr MPI_Datatype
mpi_datatype<bool>() {
  return MPI_CXX_BOOL;
};
template <>
constexpr MPI_Datatype
mpi_datatype<std::complex<float>>() {
  return MPI_CXX_COMPLEX;
};
template <>
constexpr MPI_Datatype
mpi_datatype<std::complex<double>>() {
  return MPI_CXX_DOUBLE_COMPLEX;
};
template <>
constexpr MPI_Datatype
mpi_datatype<std::complex<long double>>() {
  return MPI_CXX_LONG_DOUBLE_COMPLEX;
};

namespace GridderState {

  /** factory method
   *
   * @param comm MPI communicator with 2d topology, MPI_COMM_SELF, or
   * MPI_COMM_NULL
   * @param vis_part_index dimensional index in 2d topology of visibililty
   * partition (can be -1, if no visibility partition)
   * @param grid_part_index dimensional index in 2d topology of grid partition
   * (can be -1 if no grid partition)
   * @param device gridder device type
   * @param max_added_tasks maximum number of additional tasks (actual number
   * may be less than requested)
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
    ) noexcept;

  /** factory method
   *
   * @param comm MPI communicator
   * @param vis_part_size size of visibility partition (i.e, number of subsets
   * in partition)
   * @param grid_part_size size of grid partition (i.e, number of subsets in
   * partition)
   * @param device gridder device type
   * @param max_added_tasks maximum number of additional tasks (actual number
   * may be less than requested)
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
    ) noexcept;

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
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
    ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create(
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
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        );
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
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
    ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create2d(
        comm,
        vis_part_size,
        grid_part_size,
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        );
  }
} // end namespace GridderState

namespace Gridder {

  /** Gridder factory method
   *
   * Does not throw an exception if device argument names an unsupported device
   */
  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create(
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
    ) noexcept;

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
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
      ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create(
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
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
      );
  }

  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create2d(
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
    ) noexcept;

  template <unsigned N>
  rval_t<std::tuple<::hpg::Gridder, bool, bool>>
  create2d(
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
    const std::vector<std::array<int, size_t(N)>>& mueller_indexes,
    const std::vector<std::array<int, size_t(N)>>& conjugate_mueller_indexes
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    , const std::array<unsigned, 4>& implementation_versions =
    ::hpg::GridderState::default_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
    ) noexcept {

    return
      create2d(
        comm,
        vis_part_size,
        grid_part_size,
        device,
        max_added_tasks,
        visibility_batch_size,
        max_avg_channels_per_vis,
        init_cf_shape,
        grid_size,
        grid_scale,
        IArrayVector(mueller_indexes),
        IArrayVector(conjugate_mueller_indexes)
#ifdef HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        , implementation_versions
#endif // HPG_ENABLE_EXPERIMENTAL_IMPLEMENTATIONS
        );
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
