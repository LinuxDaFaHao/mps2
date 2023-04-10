// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-10
*
* Description: GraceQ/MPS2 project. Two-site update finite size vMPS with MPI Parallel, slave nodes.
*/

#ifndef GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPL_SLAVE_H
#define GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPL_SLAVE_H
#include <stdlib.h>
#include "gqten/gqten.h"
#include "gqmps2/algorithm/lanczos_solver.h"                        //LanczosParams
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"
#include "boost/mpi.hpp"                                            //boost::mpi
#include "gqmps2/algo_mpi/framework.h"                              //VMPSORDER
#include "gqmps2/algo_mpi/env_tensor_update_slave.h"               //MasterGrowLeftEnvironment, MasterGrowRightEnvironment
#include "gqmps2/algo_mpi/vmps/vmps_mpi_init_slave.h"                     //InitEnvsSlave
//#include "gqmps2/algo_mpi/vmps/vmps_mpi_init_master.h"                     //MPI vmps initial
#include "gqmps2/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"   //TwoSiteMPIVMPSSweepParams
#include "gqmps2/algo_mpi/lanczos_solver_mpi.h"
namespace gqmps2 {
using namespace gqten;
namespace mpi = boost::mpi;

//forward declaration
template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSRightMovingExpand(
    const std::vector<GQTensor<TenElemT, QNT> *> &,
    boost::mpi::communicator &
);

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSLeftMovingExpand(
    const std::vector<GQTensor<TenElemT, QNT> *> &,
    boost::mpi::communicator &
);

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPS(
    mpi::communicator &world
) {
  using TenT = GQTensor<TenElemT, QNT>;

  //global variables, and please careful the memory controlling for these variables.
  std::vector<TenT *> eff_ham(two_site_eff_ham_size);

  VMPS_ORDER order = program_start;
  while (order != program_final) {
    order = SlaveGetBroadcastOrder(world);
    switch (order) {
      case program_start:
        world.send(kMasterRank, 2 * world.rank(), world.rank());
        break;
      case init_grow_env:
        InitEnvsSlave<TenElemT, QNT>(world);
        break;
      case lanczos: {
        eff_ham = SlaveLanczosSolver<TenT>(world);
      }
        break;
      case svd: {
        MPISVDSlave<TenElemT>(world);
      }
        break;
      case contract_for_right_moving_expansion: {//dir='r'
        SlaveTwoSiteFiniteVMPSRightMovingExpand(eff_ham, world);
      }
        break;
      case contract_for_left_moving_expansion: {//dir='l'
        SlaveTwoSiteFiniteVMPSLeftMovingExpand(eff_ham, world);
      }
        break;
      case growing_left_env: {
        SlaveGrowLeftEnvironment(*eff_ham[0], *eff_ham[1], world);
        for (size_t i = 0; i < two_site_eff_ham_size; i++) {
          delete eff_ham[i];
        }
      }
        break;
      case growing_right_env: {
        SlaveGrowRightEnvironment(*eff_ham[3], *eff_ham[2], world);
        for (size_t i = 0; i < two_site_eff_ham_size; i++) {
          delete eff_ham[i];
        }
      }
        break;
      case program_final:std::cout << "Node" << world.rank() << " will stop." << std::endl;
        break;
      default:std::cout << "Node " << world.rank() << " cannot understand the order " << order << std::endl;
        break;
    }
  }

}

}

#endif //GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPL_SLAVE_H
