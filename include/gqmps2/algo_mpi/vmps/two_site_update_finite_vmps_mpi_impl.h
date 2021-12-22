// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-11
*
* Description: GraceQ/MPS2 project. Two-site update finite size vMPS with MPI Paralization
*/

/**
@file two_site_update_finite_vmps_mpi.h
@brief Two-site update finite size vMPS with MPI Paralization
*/

#ifndef GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPLY_H
#define GQMPS2_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPLY_H

#include <stdlib.h>
#include "gqten/gqten.h"
#include "gqmps2/algorithm/lanczos_solver.h"                        //LanczosParams
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"
#include "boost/mpi.hpp"                                            //boost::mpi
#include "gqmps2/algo_mpi/framework.h"                              //VMPSORDER
#include "gqmps2/algo_mpi/vmps/vmps_mpi_init.h"                     //MPI vmps initial
#include "gqmps2/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"   //TwoSiteMPIVMPSSweepParams
#include "gqmps2/algo_mpi/lanczos_solver_mpi.h"                     //MPI Lanczos solver            

namespace gqmps2{
using namespace gqten;

namespace mpi = boost::mpi;
//forward decelaration

template <typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPS(mpi::communicator& world);

template <typename TenElemT, typename QNT>
void LoadRelatedTensOnTwoSiteAlgWhenRightMoving(
    FiniteMPS<TenElemT, QNT> &,
    TenVec<GQTensor<TenElemT, QNT>> &,
    TenVec<GQTensor<TenElemT, QNT>> &,
    const size_t,   const size_t,
    const TwoSiteMPIVMPSSweepParams &
);

template <typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSRightMovingExpand(
  const std::vector< GQTensor<TenElemT, QNT> *> &,
  boost::mpi::communicator&
);

template <typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSLeftMovingExpand(
  const std::vector< GQTensor<TenElemT, QNT> *> &,
  boost::mpi::communicator&
);

/**
Function to perform two-site update finite vMPS algorithm with MPI paralization.
  
  @example 
  Using the API in the following way:
  in `main()`, codes like below are needed at start:
  ```
      namespace mpi = boost::mpi;
      mpi::environment env(mpi::threading::multiple);
      if(env.thread_level() < mpi::threading::multiple){
        std::cout << "thread level of env is not right." << std::endl;
        env.abort(-1);
      }
      mpi::communicator world;
  ```
  Note that multithreads environment are used to accelerate communications.
  
  When calling the function, just call it in all of the processors. No if
  condition sentences are needed.
  ```
    double e0 = TwoSiteFiniteVMPS(mps, mpo, sweep_params, world);
  ```
  However, except `world`, only variables in master processor
  (rank 0 processor) are valid, inputs of other processor(s) can be 
  arbitrary (Of course the types should be right). Outputs of slave(s)
  are all 0.0. 

  @note  The input MPS will be considered an empty one.
         The true data has be writed into disk.
  @note  The canonical center of input MPS should be set <=left_boundary+1.
        The canonical center of output MPS will move to left_boundary+1.
*/
template <typename TenElemT, typename QNT>
inline GQTEN_Double TwoSiteFiniteVMPS(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const TwoSiteMPIVMPSSweepParams &sweep_params,
    mpi::communicator& world
){
  GQTEN_Double e0(0.0);
  if(world.rank()== kMasterRank){
    e0 = MasterTwoSiteFiniteVMPS(mps,mpo,sweep_params,world);
  }else{
    SlaveTwoSiteFiniteVMPS<TenElemT, QNT>(world);
  }
  return e0;
}



template <typename TenElemT, typename QNT>
GQTEN_Double MasterTwoSiteFiniteVMPS(
  FiniteMPS<TenElemT, QNT> &mps,
  const MPO<GQTensor<TenElemT, QNT>> &mpo,
  const TwoSiteMPIVMPSSweepParams &sweep_params,
  mpi::communicator world
) {
  assert(world.rank() == kMasterRank ); //only master can call this function
  assert(mps.size() == mpo.size());
  
  MasterBroadcastOrder(program_start, world );
  auto [left_boundary, right_boundary]=TwoSiteFiniteVMPSInit(mps,mpo,sweep_params,world);
  double e0(0.0);
  mps.LoadTen(left_boundary+1, GenMPSTenName(sweep_params.mps_path, left_boundary+1));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params, 
                              left_boundary, right_boundary, world);
    
    
    sweep_timer.PrintElapsed();
    std::cout << std::endl;
  }
  mps.DumpTen(left_boundary+1, GenMPSTenName(sweep_params.mps_path, left_boundary+1), true);
  MasterBroadcastOrder(program_final, world);
  return e0;
}

template <typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPS(
  mpi::communicator& world
){
  using TenT = GQTensor<TenElemT, QNT>;

  //global variables, and please careful the memory controlling for these variables.
  std::vector< TenT *> eff_ham(two_site_eff_ham_size);
                
  VMPS_ORDER order = program_start;
  while(order != program_final ){
    order = SlaveGetBroadcastOrder(world);
    switch(order){
      case program_start:
        std::cout << "Slave " << world.rank() << " receive program start order." << std::endl;
        break;
      case lanczos:{
        eff_ham = SlaveLanczosSolver<TenT>(world);
      } break;
      case svd:{
        MPISVDSlave<TenElemT>(world);
      } break;
      case contract_for_right_moving_expansion:{//dir='r'
        SlaveTwoSiteFiniteVMPSRightMovingExpand(eff_ham, world);
      }break;
      case contract_for_left_moving_expansion:{//dir='l'
        SlaveTwoSiteFiniteVMPSLeftMovingExpand(eff_ham, world);
      }break;
      case growing_left_env:{
        SlaveGrowLeftEnvironment(*eff_ham[0], *eff_ham[1], world);
        for(size_t i=0;i<two_site_eff_ham_size;i++){
          delete eff_ham[i];
        }
      } break;
      case growing_right_env:{
        SlaveGrowRightEnvironment(*eff_ham[3],*eff_ham[2], world);
        for(size_t i=0;i<two_site_eff_ham_size;i++){
          delete eff_ham[i];
        }
      } break;
      case program_final:
        std::cout << "Slave" << world.rank() << " will stop." << std::endl;
        break;
      default:
        std::cout << "Slave " << world.rank() << " doesn't understand the order " << order << std::endl;
        break;
    }
  }

}


/**
Function to perform a single two-site finite vMPS sweep.

@note Before the sweep and after the sweep, the MPS only contains mps[1].
*/
template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const TwoSiteMPIVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    mpi::communicator world
) {
  auto N = mps.size();
  using TenT = GQTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);
  double e0;

  for (size_t i = left_boundary; i <= right_boundary - 2; ++i) {
    // Load to-be-used tensors
    LoadRelatedTensOnTwoSiteAlgWhenRightMoving(mps, lenvs, renvs, i, left_boundary, sweep_params);
    e0 = MasterTwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'r', i,world);
    // Dump related tensor to HD and remove unused tensor from RAM
    DumpRelatedTensOnTwoSiteAlgWhenRightMoving(mps, lenvs, renvs, i, sweep_params);
  }
  for (size_t i = right_boundary; i >= left_boundary+2; --i) {
    LoadRelatedTensOnTwoSiteAlgWhenLeftMoving(mps, lenvs, renvs, i, right_boundary, sweep_params);
    e0 = MasterTwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'l', i,world);
    DumpRelatedTensOnTwoSiteAlgWhenLeftMoving(mps, lenvs, renvs, i, sweep_params);
  }
  return e0;
}


template <typename TenElemT, typename QNT>
double MasterTwoSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const TwoSiteMPIVMPSSweepParams &sweep_params,
    const char dir,
    const size_t target_site,
    mpi::communicator world
) {
  //master
  Timer update_timer("two_site_fvmps_update");
#ifdef GQMPS2_TIMING_MODE
  Timer initialize_timer("two_site_fvmps_setup_and_initial_state");
#endif
  // Assign some parameters
  auto N = mps.size();
  std::vector<std::vector<size_t>> init_state_ctrct_axes;
  size_t svd_ldims;
  size_t lsite_idx, rsite_idx;
  size_t lenv_len, renv_len;
  std::string lblock_file, rblock_file;
  init_state_ctrct_axes = {{2}, {0}};
  svd_ldims = 2;
  switch (dir) {
    case 'r':
      lsite_idx = target_site;
      rsite_idx = target_site + 1;
      lenv_len = target_site;
      renv_len = N - (target_site + 2);
      break;
    case 'l':
      lsite_idx = target_site - 1;
      rsite_idx = target_site;
      lenv_len = target_site - 1;
      renv_len = N - target_site - 1;
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(1);
  }

  // Lanczos
  using TenT = GQTensor<TenElemT, QNT>;
  std::vector<TenT *>eff_ham(4);
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[lsite_idx]);
  eff_ham[2] = const_cast<TenT *>(&mpo[rsite_idx]);
  eff_ham[3] = renvs(renv_len);
  

  auto init_state = new TenT;
  Contract(&mps[lsite_idx], &mps[rsite_idx], init_state_ctrct_axes, init_state);
#ifdef GQMPS2_TIMING_MODE
  initialize_timer.PrintElapsed();
#endif
  Timer lancz_timer("two_site_fvmps_lancz");
  MasterBroadcastOrder(lanczos, world);
  auto lancz_res = MasterLanczosSolver(
                       eff_ham, init_state,
                       sweep_params.lancz_params,
                       world
                   );
#ifdef GQMPS2_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

  // SVD and measure entanglement entropy
#ifdef GQMPS2_TIMING_MODE
  Timer svd_timer("two_site_fvmps_svd");
#endif

  TenT u, vt;
  using DTenT = GQTensor<GQTEN_Double, QNT>;
  DTenT s;
  GQTEN_Double actual_trunc_err;
  size_t D;
  MasterBroadcastOrder(svd, world);
  MPISVDMaster(
      lancz_res.gs_vec,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D,
      world
  );
  delete lancz_res.gs_vec;
  auto ee = MeasureEE(s, D);

#ifdef GQMPS2_TIMING_MODE
  svd_timer.PrintElapsed();
#endif

  // Update MPS local tensor
#ifdef GQMPS2_TIMING_MODE
  Timer update_mps_ten_timer("two_site_fvmps_update_mps_ten");
#endif

  TenT the_other_mps_ten;
  switch (dir) {
    case 'r':
      mps[lsite_idx] = std::move(u);
      Contract(&s, &vt, {{1}, {0}}, &the_other_mps_ten);
      mps[rsite_idx] = std::move(the_other_mps_ten);
      break;
    case 'l':
      Contract(&u, &s, {{2}, {0}}, &the_other_mps_ten);
      mps[lsite_idx] = std::move(the_other_mps_ten);
      mps[rsite_idx] = std::move(vt);
      break;
    default:
      assert(false);
  }

#ifdef GQMPS2_TIMING_MODE
  update_mps_ten_timer.PrintElapsed();
#endif

  // Update environment tensors
#ifdef GQMPS2_TIMING_MODE
  Timer update_env_ten_timer("two_site_fvmps_update_env_ten");
#endif
  switch (dir) {
    case 'r':{
      MasterBroadcastOrder(growing_left_env, world);
      lenvs(lenv_len + 1) = MasterGrowLeftEnvironment(lenvs[lenv_len], mpo[target_site],mps[target_site], world);
      /*
      TenT temp1, temp2, lenv_ten;
      Contract(&lenvs[lenv_len], &mps[target_site], {{0}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{0, 2}, {0, 1}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{0 ,2}, {0, 1}}, &lenv_ten);
      lenvs[lenv_len + 1] = std::move(lenv_ten);
      */
    }break;
    case 'l':{
      MasterBroadcastOrder(growing_right_env, world);
      renvs(renv_len + 1) = MasterGrowRightEnvironment(*eff_ham[3], mpo[target_site],mps[target_site], world);
      /*
      TenT temp1, temp2, renv_ten;
      Contract(&mps[target_site], eff_ham[3], {{2}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{1, 2}, {1, 3}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv_ten);
      renvs[renv_len + 1] = std::move(renv_ten);
      */
    }break;
    default:
      assert(false);
  }

#ifdef GQMPS2_TIMING_MODE
  update_env_ten_timer.PrintElapsed();
#endif

  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site " << std::setw(4) << target_site
            << " E0 = " << std::setw(20) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed << lancz_res.gs_eng
            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
            << " D = " << std::setw(5) << D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}



template <typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenRightMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const size_t left_boundary,
    const TwoSiteMPIVMPSSweepParams &sweep_params
) {
#ifdef GQMPS2_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
auto N = mps.size();
if (target_site != left_boundary) {
  mps.LoadTen(
      target_site + 1,
      GenMPSTenName(sweep_params.mps_path, target_site + 1)
  );
  auto renv_len = N - (target_site + 2);
  auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
  renvs.LoadTen(renv_len, renv_file);
  RemoveFile(renv_file);
} else {
  mps.LoadTen(
      target_site,
      GenMPSTenName(sweep_params.mps_path, target_site)
  );
  auto renv_len = (N - 1) - (target_site + 1);
  auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
  renvs.LoadTen(renv_len, renv_file);
  RemoveFile(renv_file);
  auto lenv_len = target_site;
  auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
  lenvs.LoadTen(lenv_len, lenv_file);
}  
#ifdef GQMPS2_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}


template <typename TenElemT, typename QNT>
inline void LoadRelatedTensOnTwoSiteAlgWhenLeftMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const size_t right_boundary,
    const TwoSiteMPIVMPSSweepParams &sweep_params
){
#ifdef GQMPS2_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
const size_t N = mps.size();
if (target_site != right_boundary) {
  mps.LoadTen(
      target_site - 1,
      GenMPSTenName(sweep_params.mps_path, target_site - 1)
  );
  auto lenv_len = (target_site+1) - 2;
  auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
  lenvs.LoadTen(lenv_len, lenv_file);
  RemoveFile(lenv_file);
} else {
  mps.LoadTen(
      target_site,
      GenMPSTenName(sweep_params.mps_path, target_site)
  );
  auto renv_len = (N-1)-target_site;
  auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
  renvs.LoadTen(renv_len, renv_file);
  auto lenv_len = target_site - 1;
  auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
  RemoveFile(lenv_file);
}
#ifdef GQMPS2_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}

template <typename TenElemT, typename QNT>
inline void DumpRelatedTensOnTwoSiteAlgWhenRightMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const TwoSiteMPIVMPSSweepParams &sweep_params
){
#ifdef GQMPS2_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_postprocessing");
#endif
auto N = mps.size();
lenvs.dealloc(target_site);
renvs.dealloc(N - (target_site + 2));
mps.DumpTen(
    target_site,
    GenMPSTenName(sweep_params.mps_path, target_site),
    true
);
lenvs.DumpTen(
    target_site + 1,
    GenEnvTenName("l", target_site + 1, sweep_params.temp_path)
);
#ifdef GQMPS2_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}

template <typename TenElemT, typename QNT>
inline void DumpRelatedTensOnTwoSiteAlgWhenLeftMoving(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const TwoSiteMPIVMPSSweepParams &sweep_params
){
#ifdef GQMPS2_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_postprocessing");
#endif
auto N = mps.size();
lenvs.dealloc((target_site+1) - 2);
renvs.dealloc(N - (target_site+1));
mps.DumpTen(
    target_site,
    GenMPSTenName(sweep_params.mps_path, target_site),
    true
);
auto next_renv_len = N - target_site;
renvs.DumpTen(
    next_renv_len,
    GenEnvTenName("r", next_renv_len, sweep_params.temp_path)
);

#ifdef GQMPS2_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}

}//gqmps2

#endif