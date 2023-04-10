// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-11
*
* Description: GraceQ/MPS2 project. Basic set up for parallel VMPS and TDVP.
*/

/**
@file  framework.h
@brief  Basic set up for parallel VMPS and TDVP.
*/

#ifndef GQMPS2_ALGO_MPI_FRAMEWORK_H
#define GQMPS2_ALGO_MPI_FRAMEWORK_H

#include "gqten/gqten.h"
#include "boost/mpi.hpp"

namespace gqmps2 {
using namespace gqten;
const size_t kMasterRank = 0;

///< variational mps orders send by master
enum VMPS_ORDER {
  program_start,        ///< when vmps start
  init_grow_env,        ///< if need to grow env before the first sweep
  init_grow_env_grow,   ///< when grow env when initially growing env
  init_grow_env_finish,  ///< when the growing env works before the first sweep finished.
  lanczos,              ///< when lanczos start
  svd,                  ///< before svd
  lanczos_mat_vec,      ///< before do lanczos' matrix vector multiplication
  lanczos_finish,       ///< when lanczos finished
  contract_for_right_moving_expansion, ///< contraction and fuse index operations in expansion when right moving
  contract_for_left_moving_expansion, ///< contraction and fuse index operations in expansion when left moving
  growing_left_env,     ///< growing left environment
  growing_right_env,    ///< growing right environment
  program_final         /// when vmps finished
};

const size_t two_site_eff_ham_size = 4;
namespace mpi = boost::mpi;

inline void MasterBroadcastOrder(const VMPS_ORDER order,
                                 mpi::communicator &world) {
  mpi::broadcast(world, const_cast<VMPS_ORDER &>(order), kMasterRank);
}

inline VMPS_ORDER SlaveGetBroadcastOrder(mpi::communicator world) {
  VMPS_ORDER order;
  mpi::broadcast(world, order, kMasterRank);
  return order;
}

}//gqmps2

#endif