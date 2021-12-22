// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021/11/1.
*
* Description: GraceQ/MPS2 project. Implementation details for two site tdvp.
*/

#ifndef GQMPS2_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_H
#define GQMPS2_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_H

#include "gqmps2/consts.h"                      // kMpsPath, kRuntimeTempPath
#include "gqmps2/algorithm/lanczos_solver.h"    // LanczParams
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h" //SweepParams
#include <string>                               // string


namespace gqmps2 {
using namespace gqten;

template <typename QNT>
struct TDVPSweepParams {
  TDVPSweepParams() = default;
  TDVPSweepParams(
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
  ) : tau(tau), step(step), site_0(site_0),
      op0(op0), inst0(inst0),
      op1(op1), inst1(inst1),
      e0(e0),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      mps_path(mps_path),
      initial_mps_path(initial_mps_path),
      temp_path(temp_path),
      measure_temp_path(measure_temp_path) {}


  operator SweepParams() const {
    return SweepParams(
        step, Dmin, Dmax, trunc_err,
        lancz_params,
        mps_path, temp_path
        );
  }

  double tau;
  size_t step;
  size_t site_0;
  GQTensor<GQTEN_Complex, QNT> op0;
  GQTensor<GQTEN_Complex, QNT> inst0;
  GQTensor<GQTEN_Complex, QNT> op1;
  GQTensor<GQTEN_Complex, QNT> inst1;

  double e0;  //energy value of ground state(initial state)

  size_t Dmin;
  size_t Dmax;
  double trunc_err;


  LanczosParams lancz_params;

  // Advanced parameters
  /// Evolution MPS directory path
  std::string mps_path;

  /// Initial state
  std::string initial_mps_path;

  /// Runtime temporary files directory path
  std::string temp_path;

  std::string measure_temp_path;
};

}//gqmps2

#include "gqmps2/algorithm/tdvp/two_site_update_finite_tdvp_impl.h"


#endif //GQMPS2_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_H
