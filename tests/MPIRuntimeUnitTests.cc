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
#include "gtest/gtest.h"

template <typename E>
inline void
error_type_test(hpg::mpi::ErrorType etype) {
  std::unique_ptr<hpg::Error> err = std::make_unique<E>();
  EXPECT_EQ(err->type(), hpg::ErrorType::Other);
  ASSERT_NO_THROW(dynamic_cast<hpg::mpi::Error*>(err.get()));
  EXPECT_EQ(dynamic_cast<hpg::mpi::Error*>(err.get())->mpi_type(), etype);
}

TEST(MPIRuntime, Errors) {

  error_type_test<hpg::mpi::InvalidTopologyError>(
    hpg::mpi::ErrorType::InvalidTopology);
  error_type_test<hpg::mpi::InvalidCartesianRankError>(
    hpg::mpi::ErrorType::InvalidCartesianRank);
  error_type_test<hpg::mpi::IdenticalPartitionIndexError>(
    hpg::mpi::ErrorType::IdenticalPartitionIndex);
  error_type_test<hpg::mpi::InvalidPartitionIndexError>(
    hpg::mpi::ErrorType::InvalidPartitionIndex);
}

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
