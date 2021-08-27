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
         1. Check if D is sufficient, if there is QN(that's not dense tensor).
         2. Replace the boundary tensors with combiner if its bound dimension is not sufficient large;
            Make sure the central is in the left boundary site (Assuming the central around left side)
         3. Check if .temp exsits, if exsits, check if temp tensors are complete. 
            if one of above if is not, regenerate the environment.
         4. Generate the environment to left boundary site
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

namespace gqmps2 {
using namespace gqten;
namespace mpi = boost::mpi;

//forward declarition
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
std::pair<size_t,size_t> TwoSiteFiniteVMPSInit(
   FiniteMPS<TenElemT, QNT> &mps,
   const MPO<GQTensor<TenElemT, QNT>> &mpo,
   const TwoSiteMPIVMPSSweepParams &sweep_params,
   mpi::communicator world){
   
   assert(world.rank()==0);

   std::cout << "\n";
   std::cout << "=====> Two-Site MPI Update Sweep Parameters <=====" << "\n";
   std::cout << "MPS/MPO size: \t " << mpo.size() << "\n";
   std::cout << "The number of sweep times: \t " << sweep_params.sweeps << "\n";
   std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
   std::cout << "Cut off truncation error: \t " <<sweep_params.trunc_err << "\n";
   std::cout << "Lanczos max iterations \t" <<sweep_params.lancz_params.max_iterations << "\n";
   std::cout << "MPS path: \t" << sweep_params.mps_path << "\n";
   std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

   std::cout << "=====> Technical Parameters <=====" << "\n";
   std::cout << "The number of processors(including master): \t" << world.size() << "\n";
   std::cout << "The number of threads per processor: \t" << hp_numeric::GetTensorManipulationTotalThreads() <<"\n";

   std::cout << "====> Checking and updating boundary tensors --->" << std::endl;

   using Tensor = GQTensor<TenElemT, QNT>;

   auto [left_boundary, right_boundary] = CheckAndUpdateBoundaryMPSTensors(
      mps,
      sweep_params.mps_path,
      sweep_params.Dmax
   );


   //check bond dimension, can omp or mpi parallel
   /*
   for(size_t i = left_boundary; i < right_boundary; i++){
      
   }
   */
   std::cout << "====> Checking the environment tensors --->" << std::endl;
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

   //update the left env of left_boundary site and right env of right_boundary site
   UpdateBoundaryEnvs(mps, mpo, sweep_params.mps_path,
                     sweep_params.temp_path, left_boundary, right_boundary, 2 );
   return std::make_pair(left_boundary, right_boundary);
}


/**
 * 
 * @return left_boundary is the most left site needed to update, similar for right_boundary 
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

  //Assum the central of MPS at zero

  //Left Side
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  for(size_t i=0;i<left_middle_site;i++){
    mps.LoadTen(i+1, GenMPSTenName(mps_path, i+1));
    
    mps.Centralize(i+1);//improve here
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


}//gqmps2
#endif