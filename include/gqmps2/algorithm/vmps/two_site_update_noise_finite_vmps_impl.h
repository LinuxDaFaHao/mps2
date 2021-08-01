// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
*         Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-7-30
*
* Description: GraceQ/MPS2 project. Implementation details for noised two-site algorithm.
*/

/**
@file two_site_update_noise_finite_vmps_impl.h
@brief Implementation details for noised two-site algorithm.
*/

#pragma once


#include "gqmps2/algorithm/vmps/single_site_update_finite_vmps.h"   // SingleVMPSSweepParams
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"      // helper functions
#include "gqmps2/one_dim_tn/mpo/mpo.h"                              // MPO
#include "gqmps2/one_dim_tn/mps/finite_mps/finite_mps.h"            // FiniteMPS
#include "gqmps2/utilities.h"                                       // IsPathExist, CreatPath
#include "gqmps2/one_dim_tn/framework/ten_vec.h"                    // TenVec
#include "gqmps2/consts.h"
#include "gqten/gqten.h"
#include "gqten/utility/timer.h"                                    // Timer

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <stdio.h>    // remove
#ifdef Release
#define NDEBUG
#endif
#include <assert.h>

namespace gqmps2{
using namespace gqten;


using TwoSiteVMPSSweepParams = SingleVMPSSweepParams;

// Forward declarition
template <typename DTenT>
inline double MeasureEE(const DTenT &s, const size_t sdim);

template <typename TenElemT, typename QNT>
std::pair<size_t,size_t> CheckAndUpdateBoundaryMPSTensors(FiniteMPS<TenElemT, QNT> &,
                                                            const std::string&,
                                                            const size_t);

template <typename TenElemT, typename QNT>
void UpdateBoundaryEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const size_t update_site_num = 2 //e.g., two site update or single site update
);

template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(//also a overload
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const TwoSiteVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    double& noise_start
);



/**
 Function to perform two-site noised update finite vMPS algorithm.

 @note The input MPS will be considered an empty one.
 @note The canonical center of MPS should be set at around left side
*/
template <typename TenElemT, typename QNT>
GQTEN_Double TwoSiteFiniteVMPS( //same function name, overload by class of SweepParams 
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    TwoSiteVMPSSweepParams &sweep_params
){
    assert(mps.size() == mpo.size());

    std::cout << std::endl;
    std::cout << "=====> Two-Site (Noised) Update Sweep Parameter <=====" << std::endl;
    std::cout << "MPS/MPO size: \t " << mpo.size() << std::endl;
    std::cout << "The number of sweep times: \t " << sweep_params.sweeps << std::endl;
    std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << std::endl;
    std::cout << "Cut off truncation error: \t " <<sweep_params.trunc_err << std::endl;
    std::cout << "Lanczos max iterations \t" <<sweep_params.lancz_params.max_iterations << std::endl;
    std::cout << "Preseted noises: \t[";
    for(size_t i = 0; i < sweep_params.noises.size(); i++){
      std::cout << sweep_params.noises[i];
      if (i!=sweep_params.noises.size()-1) {
        std::cout << ", ";
      } else {
        std::cout << "]" << std::endl;
      }
    }
    std::cout << "MPS path: \t" << sweep_params.mps_path << std::endl;
    std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

    std::cout << "==>Checking and updating boundary tensors" << std::endl;
    auto [left_boundary, right_boundary] = CheckAndUpdateBoundaryMPSTensors(mps, sweep_params.mps_path, sweep_params.Dmax);

  
    // If the runtime temporary directory does not exit, create it and initialize
    // the right environments
    if (!IsPathExist(sweep_params.temp_path)) {
      CreatPath(sweep_params.temp_path);
      InitEnvs(mps, mpo, sweep_params.mps_path, sweep_params.temp_path, left_boundary+2 );
      std::cout << "no exsiting path " <<sweep_params.temp_path
                << ", thus progress created it and generated environment tensors."
                << std::endl;
    } else {
      std::cout << "finded exsiting path "<<sweep_params.temp_path
                << ", thus progress will use the present environment tensors."
                << std::endl;
    }
    UpdateBoundaryEnvs(mps, mpo, sweep_params.mps_path,
                        sweep_params.temp_path, left_boundary, right_boundary, 2 );
    GQTEN_Double e0;

    
    if (sweep_params.noises.size() == 0) { sweep_params.noises.push_back(0.0); }
    double noise_start;
    mps.LoadTen(left_boundary, GenMPSTenName(sweep_params.mps_path, left_boundary));
    mps.LoadTen(left_boundary+1, GenMPSTenName(sweep_params.mps_path, left_boundary+1));
    for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
      if ((sweep - 1) < sweep_params.noises.size()) {
        noise_start = sweep_params.noises[sweep-1];
      }
      std::cout << "sweep " << sweep << std::endl;
      Timer sweep_timer("sweep");
      e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params, 
                                left_boundary, right_boundary, noise_start);
      sweep_timer.PrintElapsed();
      std::cout << std::endl;
    }
    mps.DumpTen(left_boundary, GenMPSTenName(sweep_params.mps_path, left_boundary), true);
    mps.DumpTen(left_boundary+1, GenMPSTenName(sweep_params.mps_path, left_boundary+1), true);
    return e0;
}


/**
Two-site (noised) update DMRG algorithm refer to 10.1103/PhysRevB.91.155115
*/
template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(//also a overload
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const TwoSiteVMPSSweepParams &sweep_params,
    const size_t left_boundary,
    const size_t right_boundary,
    double& noise_start
) {
  auto N = mps.size();
  using TenT = GQTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N), renvs(N);
  double e0(0.0), actual_e0(0.0), actual_laststep_e0(0.0);

  const double alpha = sweep_params.alpha;
  const double noise_decrease = sweep_params.noise_decrease;
  const double noise_increase = sweep_params.noise_increase;
  const double max_noise = sweep_params.max_noise;

  double& noise_running = noise_start;
  for (size_t i = left_boundary; i < right_boundary-1; ++i) {
    //The last two site [right_boudary-1, right_boundary] will update when sweep back
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params, left_boundary);    // note: here we need mps[i](do not need load),
                                                                            // mps[i+1](do not need load), mps[i+2](need load)
                                                                            // lenvs[i](do not need load), and mps[i+1]'s renvs
                                                                            // mps[i+1]'s renvs file can be removed
    actual_e0 = CalEnergyEptTwoSite(mps, mpo,lenvs, renvs, i, i+1);
    if ((actual_e0 - e0) <= 0.0) {
      // expand and truncate let the energy lower or not change
      // this case is very rare, but include the boundary mps tensor case
      // so we do nothing now
    } else if ((actual_e0 - e0) >= alpha*fabs(actual_laststep_e0-e0)) {
      // below two case suppose actual_laststep_e0-laststep_e0>0, usually it is right
      noise_running = noise_running*noise_decrease;
    } else {
      noise_running = std::min(noise_running*noise_increase, max_noise);
    }
    e0 = TwoSiteFiniteVMPSUpdate(
             mps,
             lenvs, renvs,
             mpo,
             sweep_params, 'r', i,
             noise_running
         );
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params);    // note: here we need dump mps[i](free memory),
                                                                              // lenvs[i+1](without free memory)
  }

  for (size_t i = right_boundary; i > left_boundary+1; --i) {
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params, right_boundary);
    actual_e0 = CalEnergyEptTwoSite(mps, mpo,lenvs, renvs, i-1,i);
    if ((actual_e0 - e0) <= 0.0) {
    } else if ((actual_e0 - e0) >= alpha*fabs(actual_laststep_e0 - e0)) {
      noise_running = noise_running * noise_decrease;
    } else {
      noise_running = std::min(noise_running * noise_increase, max_noise);
    }
    e0 = TwoSiteFiniteVMPSUpdate(
             mps,
             lenvs, renvs,
             mpo,
             sweep_params, 'l', i,
             noise_running
         );
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
  }
  return e0;
}


/**  Single step for two site noised update.
This function includes below procedure:
- update `mps[target]` and `mps[next_site]` tensors according corresponding environment tensors and the mpo tensor, using lanczos algorithm;
- expand `mps[target]*mps[next_site]` and `mps[next_next_site]` by noise, if need;
- canonicalize mps to `mps[next_site]` by SVD, while truncate tensor `mps[target]` if need;
- generate the next environment in the direction.

When using this function, one must make sure memory at least contains `mps[target]` tensor,
`mps[next_site]`, its environment tensors, and `mps[next_next_site]`
*/
template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const TwoSiteVMPSSweepParams &sweep_params,
    const char dir,
    const size_t target_site,
    const double preset_noise
) {
  Timer update_timer("two_site_fvmps_update");

#ifdef GQMPS2_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
  double noise = preset_noise;
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
   preprocessing_timer.PrintElapsed();
#endif
    // Lanczos
  Timer lancz_timer("two_site_fvmps_lancz");
  auto lancz_res = LanczosSolver(
                       eff_ham, init_state,
                       &eff_ham_mul_two_site_state,
                       sweep_params.lancz_params
                   );//Note here init_state is deleted
#ifdef GQMPS2_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif


#ifdef GQMPS2_TIMING_MODE
  Timer expand_timer("two_site_fvmps_expand");
#endif

  bool need_expand(true);
  if (fabs(noise) < 1e-10) {
    noise = 0.0;
    need_expand = false;
  } else if (false // QN cover??
    ) {
    noise = 0.0;            //just for output
    need_expand= false;
  }

  if (need_expand) {
    TwoSiteFiniteVMPSExpand(
        mps,
        lancz_res.gs_vec,
        eff_ham,
        dir,
        target_site,
        noise
    );
  }

#ifdef GQMPS2_TIMING_MODE
  expand_timer.PrintElapsed();
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
  SVD(
      lancz_res.gs_vec,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D
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
      TenT temp1, temp2, lenv_ten;
      Contract(&lenvs[lenv_len], &mps[target_site], {{0}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{0, 2}, {0, 1}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{0 ,2}, {0, 1}}, &lenv_ten);
      lenvs[lenv_len + 1] = std::move(lenv_ten);
    }break;
    case 'l':{
      TenT temp1, temp2, renv_ten;
      Contract(&mps[target_site], eff_ham[3], {{2}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{1, 2}, {1, 3}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv_ten);
      renvs[renv_len + 1] = std::move(renv_ten);
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
            << " noise = " <<  std::setprecision(2) << std::scientific  << noise << std::fixed
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
void TwoSiteFiniteVMPSExpand(
    FiniteMPS<TenElemT, QNT> &mps,
    GQTensor<TenElemT, QNT> *gs_vec,
    const std::vector< GQTensor<TenElemT, QNT> *> &eff_ham,
    const char dir,
    const size_t target_site,
    const double noise
) {
  // note: The expanded tensors are saved in *gs_vec, and mps[next_next_site]
  using TenT = GQTensor<TenElemT, QNT>;
  TenT* ten_tmp = new TenT();
  // we suppose mps contain mps[targe_site], mps[next_site],  mps[next_next_site]
  if (dir=='r') {
    Contract(eff_ham[0], gs_vec, {{0}, {0}}, ten_tmp);
    InplaceContract(ten_tmp, eff_ham[1], {{0, 2}, {0, 1}});
    InplaceContract(ten_tmp, eff_ham[2], {{4, 1}, {0, 1}});
    ten_tmp->FuseIndex(1, 4);
    (*ten_tmp) *= noise;
    gs_vec->Transpose({3,0,1,2});
    TenT expanded_ten;
    ExpandMC(gs_vec, ten_tmp, {0},  &expanded_ten);
    expanded_ten.Transpose({1,2,3,0});
    (*gs_vec) = std::move(expanded_ten);
    
    size_t next_next_site = target_site + 2;
    auto expanded_index = InverseIndex(ten_tmp->GetIndexes()[0]);
    TenT expanded_zero_ten = TenT({
                                 expanded_index,
                                 mps[next_next_site].GetIndexes()[1],
                                 mps[next_next_site].GetIndexes()[2]
                             });
    (*ten_tmp) = TenT();
    ExpandMC(mps(next_next_site), &expanded_zero_ten, {0}, ten_tmp);
    delete mps(next_next_site);
    mps(next_next_site) = ten_tmp;
  } else if (dir=='l') {
    size_t next_next_site = target_site - 2;
    Contract(gs_vec, eff_ham[3], {{3}, {0}}, ten_tmp);

    InplaceContract(ten_tmp, eff_ham[2], {{2,3}, {1, 3}});
    InplaceContract(ten_tmp, eff_ham[1], {{1,3},{1,3}});
    ten_tmp->Transpose({0, 3,4,2,1});
    ten_tmp->FuseIndex(0,1);
    (*ten_tmp) *= noise;
    TenT expanded_ten;
    ExpandMC(gs_vec, ten_tmp, {0}, &expanded_ten);
    *gs_vec = std::move(expanded_ten);
    

    auto expanded_index = InverseIndex(ten_tmp->GetIndexes()[0]);
    TenT expanded_zero_ten = TenT({
                                 mps[next_next_site].GetIndexes()[0],
                                 mps[next_next_site].GetIndexes()[1],
                                 expanded_index
                             });
    *ten_tmp = TenT();
    ExpandMC(mps(next_next_site), &expanded_zero_ten, {2}, ten_tmp);
    delete mps(next_next_site);
    mps(next_next_site) = ten_tmp;
  }
}


/**
 * 
 * 
 * @note the central of mps will be moved to left_boundary+1.
*/
template <typename TenElemT, typename QNT>
std::pair<size_t,size_t> CheckAndUpdateBoundaryMPSTensors(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string& mps_path,
    const size_t Dmax
){
  assert(mps.empty());
  //TODO: check if central file, add this function to the friend of FiniteMPS
  using TenT = GQTensor<TenElemT, QNT>;
  
  using std::cout;
  using std::endl;
  size_t N = mps.size();
  size_t left_boundary(0);  //the most left site which needs to update.
  size_t right_boundary(0); //the most right site which needs to update
  
  size_t left_middle_site, right_middle_site;
  if(N%2==0){
    left_middle_site = N/2-1;
    right_middle_site = N/2;
    //make sure at least four sites are used to sweep
  }else{
    left_middle_site = N/2;
    right_middle_site = N/2;
    //make sure at least three sites are used to sweep
  }

  //Set the central of MPS at zero
  mps.tens_cano_type_[0] = MPSTenCanoType::NONE;
  for(size_t i=0;i<N;i++){
    mps.tens_cano_type_[i] = MPSTenCanoType::RIGHT;
  }
  mps.center_ = 0;

  //Left Side
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  for(size_t i=0;i<left_middle_site;i++){
    mps.LoadTen(i+1, GenMPSTenName(mps_path, i+1));
    mps.Centralize(i+1);
    TenT& mps_ten = mps[i];
    ShapeT mps_ten_shape = mps_ten.GetShape();
    if(mps_ten_shape[0]*mps_ten_shape[1]>Dmax ){
        left_boundary = i;
        break;
    }else if(mps_ten_shape[0]*mps_ten_shape[1]>mps_ten_shape[2]){
        GQTenIndexDirType new_dir = mps_ten.GetIndexes()[2].GetDir();
        Index<QNT> index_0 = mps_ten.GetIndexes()[0];
        Index<QNT> index_1 = mps_ten.GetIndexes()[1];

        TenT index_combiner_for_fuse = IndexCombine<TenElemT,QNT>(
                                    InverseIndex(index_0),
                                    InverseIndex(index_1),
                                    IN
        );
        TenT ten_tmp;
        Contract(&index_combiner_for_fuse, &mps_ten, {{0,1},{0,1}},&ten_tmp);
        mps_ten = std::move(ten_tmp);

        TenT index_combiner =  IndexCombine<TenElemT,QNT>(
                                index_0,
                                index_1,
                                new_dir
                                );
        
        assert(mps[i].GetIndexes()[0] == InverseIndex( index_combiner.GetIndexes()[2] ) );
        TenT mps_next_tmp;
        Contract(mps(i), mps(i+1),{{1},{0}}, &mps_next_tmp );
        mps[i+1] = std::move(mps_next_tmp);
        mps[i] = std::move(index_combiner);

        mps.center_ = i+1;
        mps.tens_cano_type_[i] = LEFT;
        mps.tens_cano_type_[i+1] = NONE;
    }
    if(i == left_middle_site-1){
        left_boundary = i;
    }
  }
  
  for(size_t i=0;i<=left_boundary+1;i++){
      mps.DumpTen(i, GenMPSTenName(mps_path, i), true);
  }
  
  //Right Side
  mps.LoadTen(N-1, GenMPSTenName(mps_path, N-1));
  for(size_t i=N-1;i>right_middle_site;i--){
    //The mps centre has been sent to the left half side
    mps.LoadTen(i-1, GenMPSTenName(mps_path, i-1));
    TenT& mps_ten = mps[i];
    ShapeT mps_ten_shape = mps_ten.GetShape();
    if(mps_ten_shape[1]*mps_ten_shape[2]>Dmax){
        right_boundary = i;
        break;
    }else if(mps_ten_shape[1]*mps_ten_shape[2]>mps_ten_shape[0]){
        TenT index_combiner = IndexCombine<TenElemT,QNT>(
                mps[i].GetIndexes()[1],
                mps[i].GetIndexes()[2],
                mps[i].GetIndexes()[0].GetDir()
                );
        index_combiner.Transpose({2,0,1});
        mps[i].FuseIndex(1,2);
        assert(mps[i].GetIndexes()[0] == InverseIndex( index_combiner.GetIndexes()[0] ) );
        InplaceContract(mps(i-1), mps(i),{{2},{1}});
        mps[i] = std::move(index_combiner);
    }

    if(i ==right_middle_site+1 ){
      right_boundary = i;
    }
  }
  for(size_t i=N-1;i>=right_boundary-1;i--){
      mps.DumpTen(i, GenMPSTenName(mps_path, i), true);
  }

  assert(mps.empty());
  return std::make_pair(left_boundary, right_boundary);
}
/**
 * Check if there is environment folder. 
 * If no, generate the folder.
 * If there is, check if there's the expected environment tensors.
 * If no, generate it.
 * If yes, check the type of the environment tensor by indexes.
 * If not match, update the environment tensor.
 * 
 * @note mps should has center, center == left_boundary+1
 * @note this design need more test. 
 * If there are some thing unexpected, delete environment folder and rerun.
*/ 
template <typename TenElemT, typename QNT>
void UpdateBoundaryEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t left_boundary,
    const size_t right_boundary,
    const size_t update_site_num //e.g., two site update or single site update
){
  assert(mps.empty());

  using TenT = GQTensor<TenElemT, QNT>;
  auto N = mps.size();

  //Write a trivial right environment tensor to disk
  mps.LoadTen(N-1, GenMPSTenName(mps_path, N-1));
  auto mps_trivial_index = mps.back().GetIndexes()[2];
  auto mpo_trivial_index_inv = InverseIndex(mpo.back().GetIndexes()[3]);
  auto mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  TenT renv = TenT({mps_trivial_index_inv, mpo_trivial_index_inv, mps_trivial_index});
  renv({0, 0, 0}) = 1;
  mps.dealloc(N-1);

  //bulk right environment tensors
  for (size_t i = 1; i <= N - right_boundary - 1; ++i) {
    mps.LoadTen(N-i, GenMPSTenName(mps_path, N-i)); 
    TenT temp1;
    Contract(&mps[N-i], &renv, {{2}, {0}}, &temp1);
    renv = TenT();
    TenT temp2;
    Contract(&temp1, &mpo[N-i], {{1, 2}, {1, 3}}, &temp2);
    auto mps_ten_dag = Dag(mps[N-i]);
    Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv);
    mps.dealloc(N-i);
  }
  std::string file = GenEnvTenName("r", N - right_boundary - 1, temp_path);
  WriteGQTensorTOFile(renv, file);



  //Write a trivial left environment tensor to disk
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  mps_trivial_index = mps.front().GetIndexes()[0];
  mpo_trivial_index_inv = InverseIndex(mpo.front().GetIndexes()[0]);
  mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  TenT lenv = TenT({mps_trivial_index_inv, mpo_trivial_index_inv, mps_trivial_index});
  lenv({0, 0, 0}) = 1;
  mps.dealloc(0);
  std::cout << "left boundary = " << left_boundary <<std::endl;
  for (size_t i = 0; i < left_boundary; ++i) {
    mps.LoadTen(i, GenMPSTenName(mps_path, i)); 
    TenT temp1;
    Contract(&mps[i], &lenv, {{0}, {0}}, &temp1);
    lenv = TenT();
    TenT temp2;
    Contract(&temp1, &mpo[i], {{0,2}, {1, 0}}, &temp2);
    auto mps_ten_dag = Dag(mps[i]);
    Contract(&temp2, &mps_ten_dag, {{1,2}, {0,1}}, &lenv);
    mps.dealloc(i);
  }
  file = GenEnvTenName("l", left_boundary, temp_path);
  WriteGQTensorTOFile(lenv, file);
  assert(mps.empty());
}




template <typename TenElemT, typename QNT>
void LoadRelatedTensTwoSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const TwoSiteVMPSSweepParams &sweep_params,
    const size_t boundary
) {
#ifdef GQMPS2_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
  auto N = mps.size();
  switch (dir) {
    case 'r':
      if (target_site == boundary) {//left_boundary
        mps.LoadTen(
            target_site+2,
            GenMPSTenName(sweep_params.mps_path, target_site+2)
        );

        auto renv_len = (N - 1) - (target_site + 1);
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);

        auto lenv_len = target_site;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        lenvs.LoadTen(lenv_len, lenv_file);
      } else {
        mps.LoadTen(
            target_site + 2,
            GenMPSTenName(sweep_params.mps_path, target_site + 2)
        );
        auto renv_len = N - (target_site + 2);
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);
      }
      break;
    case 'l':
      if (target_site == boundary) { //right_boundary
        mps.LoadTen(
            target_site-2,
            GenMPSTenName(sweep_params.mps_path, target_site-2)
        );
        auto renv_len = N-1-target_site;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);

        auto lenv_len = N-2;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        RemoveFile(lenv_file);
      } else {
        mps.LoadTen(
            target_site - 2,
            GenMPSTenName(sweep_params.mps_path, target_site - 2)
        );
        auto lenv_len = (target_site+1) - 2;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        lenvs.LoadTen(lenv_len, lenv_file);
        RemoveFile(lenv_file);
      }
      break;
    default:
      assert(false);
  }
#ifdef GQMPS2_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}


template <typename TenElemT, typename QNT>
void DumpRelatedTensTwoSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const TwoSiteVMPSSweepParams &sweep_params
) {
#ifdef GQMPS2_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_postprocessing");
#endif
  auto N = mps.size();
  switch (dir) {
    case 'r':{
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
    }break;
    case 'l':{
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
    }break;
    default:
      assert(false);
  }
#ifdef GQMPS2_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}



template <typename TenElemT, typename QNT>
double CalEnergyEptTwoSite(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t lsite,
    const size_t rsite
) {
  using TenT = GQTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(4);
  size_t lenv_len = lsite;
  size_t renv_len = mps.size() - rsite - 1;
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[lsite]);
  eff_ham[2] = const_cast<TenT *>(&mpo[rsite]);
  eff_ham[3] = renvs(renv_len);
  TenT wave_function;
  Contract(mps(lsite), mps(rsite),{{2},{0}},&wave_function);
  TenT *h_mul_state = eff_ham_mul_two_site_state(eff_ham, &wave_function);
  TenT scalar_ten;
  TenT wave_function_dag = Dag(wave_function);
  Contract(h_mul_state, &wave_function_dag, {{0, 1, 2, 3}, {0, 1, 2, 3}}, &scalar_ten);
  delete h_mul_state;
  double energy = Real(scalar_ten());
  return energy;
}
}