// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-12-21
*
* Description: GraceQ/MPS2 project. Two-site update finite size vMPS with MPI Paralization
*/

#ifndef GQMPS2_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_H
#define GQMPS2_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_H

#include "gqmps2/algorithm/tdvp/two_site_update_finite_tdvp.h"    //TDVPSweepParams

namespace gqmps2 {

template <typename QNT>
struct MPITDVPSweepParams : public TDVPSweepParams<QNT> {
  MPITDVPSweepParams() = default;
  MPITDVPSweepParams(
      const double tau, const size_t step,
      const size_t site_0,
      const GQTensor<GQTEN_Complex, QNT>& op0,
      const GQTensor<GQTEN_Complex, QNT>& inst0,
      const GQTensor<GQTEN_Complex, QNT>& op1,
      const GQTensor<GQTEN_Complex, QNT>& inst1,
      const double e0,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::string mps_path = kMpsPath,
      const std::string initial_mps_path = "initial_" + kMpsPath,
      const std::string temp_path = kRuntimeTempPath,
      const std::string measure_temp_path = ".measure_temp"
      ) : TDVPSweepParams<QNT>(
          tau, step, site_0, op0, inst0, op1, inst1, e0,
          dmin, dmax, trunc_err, lancz_params, mps_path, initial_mps_path,
          temp_path, measure_temp_path
          ) {}

};

}





#include "gqmps2/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi_impl_master.h"
#include "gqmps2/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi_impl_slave.h"
#endif //GQMPS2_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_H