// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-06 
*
* Description: GraceQ/MPS2 project. Initilization for two-site update finite size vMPS with MPI Paralization
*/

/**
 @file   two_site_update_finite_vmps_init.h
 @brief  Initilization for two-site update finite size vMPS with MPI Paralization.
         0. include an overall initial function cover all (at least most) the functions in this file;
         1. Find the left/right boundaries, only between which the tensors need to be update.
            Also make sure the bond dimensions of tensors out of boundaries are sufficient large.
            Move the centre on the left_boundary+1 site (Assuming the before the centre <= left_boundary+1)
         2. Check if .temp exsits, if exsits, check if temp tensors are complete. 
            if one of above if is not, regenerate the environment.
         3. Check if QN sector numbers are enough. (Not do, will deal in tensor contraction functions);
         4. Generate the environment of boundary tensors
         5. Optional function: check if different processors read/write the same disk
*/

#ifndef GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_INIT_H
#define GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_INIT_H


#include <map>
#include "gqten/gqten.h"
#include "gqmps2/one_dim_tn/mps_all.h"
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"
#include "gqmps2/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"
#include "boost/mpi.hpp"
#include "gqmps2/algorithm/vmps/vmps_init.h"                        // CheckAndUpdateBoundaryMPSTensors...

namespace gqmps2 {
using namespace gqten;
namespace mpi = boost::mpi;

//forward declarition
template <typename TenElemT, typename QNT>
std::pair<size_t,size_t> CheckAndUpdateBoundaryMPSTensors(
    FiniteMPS<TenElemT, QNT> &,
    const std::string&,
    const size_t
);

template <typename TenElemT, typename QNT>
void UpdateBoundaryEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const size_t update_site_num
);

inline bool NeedGenerateRightEnvs(
  const size_t N, //mps size
  const size_t left_boundary,
  const size_t right_boundary,
  const std::string& temp_path
);


template <typename TenElemT, typename QNT>
std::pair<size_t,size_t> TwoSiteFiniteVMPSInit(
  FiniteMPS<TenElemT, QNT> &mps,
  const MPO<GQTensor<TenElemT, QNT>> &mpo,
  const TwoSiteMPIVMPSSweepParams &sweep_params,
  mpi::communicator world){
  
  assert(world.rank()==0);
  std::cout << "\n";
  std::cout << "=====> Two-Site MPI Update Sweep Parameters <=====" << "\n";
  std::cout << "MPS/MPO size: \t " << mpo.size() << "\n";
  std::cout << "Sweep times: \t " << sweep_params.sweeps << "\n";
  std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
  std::cout << "Truncation error: \t " <<sweep_params.trunc_err << "\n";
  std::cout << "Lanczos max iterations \t" <<sweep_params.lancz_params.max_iterations << "\n";
  std::cout << "MPS path: \t" << sweep_params.mps_path << "\n";
  std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

  std::cout << "=====> Technical Parameters <=====" << "\n";
  std::cout << "The number of processors(including master): \t" << world.size() << "\n";
  std::cout << "The number of threads per processor: \t" << hp_numeric::GetTensorManipulationThreads() <<"\n";
  
  std::cout << "=====> Checking and updating boundary tensors =====>" << std::endl;
  using Tensor = GQTensor<TenElemT, QNT>;
  auto [left_boundary, right_boundary] = CheckAndUpdateBoundaryMPSTensors(
     mps,
     sweep_params.mps_path,
     sweep_params.Dmax
  );


   //check qumber sct numbers, > 2*slave number, can omp or mpi parallel
   // A best scheme is to write a more robust contraction
   /*
   for(size_t i = left_boundary; i < right_boundary; i++){
      
   }
   */
   
   if(NeedGenerateRightEnvs(
        mpo.size(), 
        left_boundary,
        right_boundary,
        sweep_params.temp_path )
    ){
      std::cout << "=====> Creating the environment tensors =====>" << std::endl;
      InitEnvs(mps, mpo, sweep_params.mps_path, sweep_params.temp_path, left_boundary+2 );
    }else {
      std::cout << "Found the environment tensors." << std::endl;
    }

   //update the left env of left_boundary site and right env of right_boundary site
   UpdateBoundaryEnvs(mps, mpo, sweep_params.mps_path,
                     sweep_params.temp_path, left_boundary, right_boundary, 2 );
   return std::make_pair(left_boundary, right_boundary);
}

}//gqmps2
#endif
