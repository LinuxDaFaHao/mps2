// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-06
*
* Description: GraceQ/MPS2 project. Initialize for two-site update finite size vMPS with MPI Parallel, slave side.
*/


#ifndef GQMPS2_ALGO_MPI_VMPS_VMPS_MPI_INIT_SLAVE_H
#define GQMPS2_ALGO_MPI_VMPS_VMPS_MPI_INIT_SLAVE_H

#include "gqten/gqten.h"
#include "boost/mpi.hpp"
#include "gqmps2/algo_mpi/mps_algo_order.h"
#include "gqmps2/algo_mpi/env_tensor_update_slave.h"  //SlaveGrowRightEnvironmentInit
namespace gqmps2 {
using namespace gqten;
namespace mpi = boost::mpi;
template<typename TenElemT, typename QNT>
void InitEnvsSlave(mpi::communicator& world) {
  auto order = SlaveGetBroadcastOrder(world);
  while (order != init_grow_env_finish) {
    assert(order == init_grow_env_grow);
    SlaveGrowRightEnvironmentInit<TenElemT, QNT>(world);
    order = SlaveGetBroadcastOrder(world);
  }
  return;
}
}//gqmps2

#endif //GQMPS2_ALGO_MPI_VMPS_VMPS_MPI_INIT_SLAVE_H
