// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-06 
*
* Description: GraceQ/MPS2 project. Two-site update finite size vMPS with MPI Paralization
*/

/**
@file two_site_update_finite_vmps_mpi.h
@brief Two-site update finite size vMPS with MPI Paralization
*/
#ifndef GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_H
#define GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_H

#include <stdlib.h>
#include "gqmps2/algorithm/lanczos_solver.h"
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"


namespace gqmps2 {

const double kMinBondDimensionOfMPIVMPS = 200;

struct TwoSiteMPIVMPSSweepParams {
  TwoSiteMPIVMPSSweepParams(
      const size_t sweeps,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::string mps_path = kMpsPath,
      const std::string temp_path = kRuntimeTempPath
  ) :
      sweeps(sweeps),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      mps_path(mps_path),
      temp_path(temp_path) {}

  size_t sweeps;

  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  LanczosParams lancz_params;


  // Advanced parameters
  /// MPS directory path
  std::string mps_path;

  /// Runtime temporary files directory path
  std::string temp_path;
};

}//gqmps2

#include "gqmps2/algo_mpi/vmps/two_site_update_finite_vmps_mpi_impl.h"
#endif