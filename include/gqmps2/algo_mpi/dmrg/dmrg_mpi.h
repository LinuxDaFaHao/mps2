// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-05-11
*
* Description: GraceQ/mps2 project. Two-site update finite size DMRG with MPI Parallelization
*/

#ifndef GQMPS2_ALGO_MPI_DMRG_DMRG_MPI_H
#define GQMPS2_ALGO_MPI_DMRG_DMRG_MPI_H

#include "gqten/gqten.h"
#include "dmrg_mpi_impl_master.h"
#include "dmrg_mpi_impl_slave.h"

namespace gqmps2 {
using namespace gqten;

template<typename TenElemT, typename QNT>
inline GQTEN_Double FiniteDMRG(
    FiniteMPS<TenElemT, QNT> &mps,
    const MatReprMPO<GQTensor<TenElemT, QNT>> &mat_repr_mpo,
    const FiniteVMPSSweepParams &sweep_params,
    mpi::communicator &world
) {
  GQTEN_Double e0(0.0);

  if (world.size() == 1) {
    DMRGExecutor<TenElemT, QNT> dmrg_executor = DMRGExecutor(mat_repr_mpo, sweep_params);
    dmrg_executor.Execute();
    return dmrg_executor.GetEnergy();
  }

  if (world.rank() == kMasterRank) {
    DMRGMPIMasterExecutor<TenElemT, QNT> dmrg_executor = DMRGMPIMasterExecutor(mat_repr_mpo, sweep_params, world);
    dmrg_executor.Execute();
    e0 = dmrg_executor.GetEnergy();
  } else {
    DMRGMPISlaveExecutor<TenElemT, QNT> dmrg_executor = DMRGMPISlaveExecutor(mat_repr_mpo, world);
    dmrg_executor.Execute();
  }
  return e0;
}
}

#endif //GQMPS2_ALGO_MPI_DMRG_DMRG_MPI_H
