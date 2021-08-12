// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-06 
*
* Description: GraceQ/MPS2 project. Lanczos solver based on distributed memory parallel.
*/

/**
@file lanczos_solver_mpi.h
@brief Lanczos solver based on distributed memory parallel.
*/

/**
 * For the note of mpi, we use //${number} to mark synchronous things 
 */ 

#ifndef GQMPS2_ALGO_MPI_LANCZOS_SOLVER_MPI_H
#define GQMPS2_ALGO_MPI_LANCZOS_SOLVER_MPI_H

#include <stdlib.h>     // size_t
#include "gqmps2/algorithm/lanczos_solver.h" // Lanczos Params
#include "boost/mpi.hpp"
#include "gqmps2/algo_mpi/utility_mpi.h"
#include "gqten/gqten.h"

namespace gqmps2 {
using namespace gqten;

namespace mpi = boost::mpi;
/**
Obtain the lowest energy eigenvalue and corresponding eigenstate from the effective
Hamiltonian and a initial state using Lanczos algorithm, with the help of distributed paralization.

 @param rpeff_ham Effective Hamiltonian as a vector of pointer-to-tensors.
 @param pinit_state Pointer to initial state for Lanczos iteration.
 @param params Parameters for Lanczos solver.
 @note only support two site update now.
*/
template <typename TenT>
LanczosRes<TenT> MasterLanczosSolver(
    const std::vector<TenT *> &rpeff_ham,
    TenT *pinit_state,
    const LanczosParams &params,
    mpi::communicator world
) {
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  //Broadcast eff_ham
  // MasterBroadcastOrder(lanczos_start, world);
  // MasterBroadcastEffectiveHamiltonian(eff_ham, world);
  const size_t eff_ham_size = pinit_state->Rank();//4
  for(size_t i = 0; i < eff_ham_size; i++){
    SendBroadCastGQTensor(world, (*rpeff_ham[i]), kMasterRank);
  }


  LanczosRes<TenT> lancz_res;

  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;

  energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  

  std::vector<TenT *> bases(params.max_iterations);
  std::vector<GQTEN_Double> a(params.max_iterations, 0.0);
  std::vector<GQTEN_Double> b(params.max_iterations, 0.0);
  std::vector<GQTEN_Double> N(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
  pinit_state->Normalize();
  bases[0] = pinit_state;

#ifdef GQMPS2_TIMING_MODE
  Timer mat_vec_timer("lancz_mat_vec");
#endif

  TenT* last_mat_mul_vec_res = master_two_site_eff_ham_mul_state(rpeff_ham, bases[0]);

#ifdef GQMPS2_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  TenT temp_scalar_ten;
  auto base_dag = Dag(*bases[0]);
  Contract(
      last_mat_mul_vec_res, &base_dag,
      energy_measu_ctrct_axes,
      &temp_scalar_ten
  );
  a[0] = Real(temp_scalar_ten());;
  N[0] = 0.0;
  size_t m = 0;
  GQTEN_Double energy0;
  energy0 = a[0];
  // Lanczos iterations.
  while (true) {
    m += 1;
    auto gamma = last_mat_mul_vec_res;
    if (m == 1) {
      LinearCombine({-a[m-1]}, {bases[m-1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m-1], -std::sqrt(N[m-1])},
          {bases[m-1], bases[m-2]},
          1.0,
          gamma
      );
    }
    auto norm_gamma = gamma->Normalize();
    GQTEN_Double eigval;
    GQTEN_Double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new TenT(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new TenT(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      }
    }

    N[m] = std::pow(norm_gamma, 2.0);
    b[m-1] = norm_gamma;
    bases[m] = gamma;

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif

    last_mat_mul_vec_res = master_two_site_eff_ham_mul_state(rpeff_ham, bases[m]);

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    TenT temp_scalar_ten;
    auto base_dag = Dag(*bases[m]);
    Contract(
        last_mat_mul_vec_res,
        &base_dag,
        energy_measu_ctrct_axes,
        &temp_scalar_ten
    );
    a[m] = Real(temp_scalar_ten());
    TridiagGsSolver(a, b, m+1, eigval, eigvec, 'N');
    auto energy0_new = eigval;
    if (
        ((energy0 - energy0_new) < params.error) ||
        (m == eff_ham_eff_dim) ||
        (m == params.max_iterations - 1)
    ) {
      TridiagGsSolver(a, b, m+1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new TenT(bases[0]->GetIndexes());
      LinearCombine(m+1, eigvec, bases, 0.0, gs_vec);
      lancz_res.iters = m;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, last_mat_mul_vec_res);
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}


template <typename TenT>
LanczosRes<TenT> SlaveLanczosSolver(
    mpi::communicator world
){
std::vector< TenT *> rpeff_ham(two_site_eff_ham_size);
for(size_t i=0;i<two_site_eff_ham_size;i++){
  rpeff_ham[i] = new TenT();
}
// Receive Hamiltonian
for(size_t i=0;i<two_site_eff_ham_size;i++){
  RecvBroadCastGQTensor(world, *rpeff_ham[i], kMasterRank);
}








for(size_t i=0;i<two_site_eff_ham_size;i++){
  delete rpeff_ham[i];
}
}


/**
 * two site effective hamiltonian multiplying on state, the dispatch works of master.
 * 
 * @param eff_ham   effective hamiltonian
 * @param state     wave function
 * @param world     boost::mpi::communicator
 * 
 * @return the result of effective hamiltonian multiple 
 */
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> master_two_site_eff_ham_mul_state(
    const std::vector<GQTensor<ElemT, QNT> *> &eff_ham,
    GQTensor<ElemT, QNT> *state,
    mpi::communicator world
) {
  using TenT = GQTensor<ElemT, QNT>;
#ifdef GQMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer(" broadcast_state_send");
#endif
  SendBroadCastGQTensor(world, *state, kMasterRank);
#ifdef GQMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  //prepare
  const size_t split_idx = 2;
  const Index<QNT>& splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();//total task number
  std::vector<TenT> res_list;
  res_list.reserve(task_num);
  const size_t slave_num = world.size() - 1 ; //total number of slaves
  //$1


  //if task_num > slave_num:
  //below for loop dispatch tasks from slave_num to task_num-1,
  //from task_num to (slave_num+task_num-1), master informs every slave that the jobs are finished.

  //if task_num <= slave_num, master informs every working slave that the jobs are finished.
  //$2
  for(size_t task = slave_num; task < slave_num+task_num ; task++){
    res_list.push_back( TenT() );
    mpi::status recv_status=recv_gqten(world, mpi::any_source, mpi::any_tag, res_list.back());
    int slave_identifier = recv_status.source();
    world.send(slave_identifier, 2*slave_identifier, task);
  }

  TenT res;
  CollectiveLinearCombine(res_list, res);
  return res;
}


/**
 * two site effective hamiltonian multiplying on state,
 * split index contract tasks worked on slave. The works are controlled by master.
 * 
 * every slave should have a copy of all of the eff_ham before call this function.
 * slave do not need prepare the state.
 * 
 * @param eff_ham   effective hamiltonian
 * @param state     wave function
 * @param world     boost::mpi::communicator
 * 
 * @return the result of effective hamiltonian multiple 
 */
template <typename ElemT, typename QNT>
void slave_two_site_eff_ham_mul_state(
    const std::vector<GQTensor<ElemT, QNT> *> &eff_ham,
    mpi::communicator world
){
  using TenT = GQTensor<ElemT, QNT>;
  TenT* state = new TenT();
#ifdef GQMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer(" broadcast_state_recv");
#endif
  RecvBroadCastGQTensor(world, *state, kMasterRank);
#ifdef GQMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif

  const size_t split_idx = 2;
  const Index<QNT>& splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();
  //slave also need to know the total task number used to judge if finish this works

  const size_t slave_identifier = world.rank();//number from 1
  if(slave_identifier > task_num){
    //no task, happy~
    delete state;
    return;
  }
  //$1

  //first task
  size_t task = slave_identifier-1;
  TenT temp1, temp2, res;
  Contract1Sector(eff_ham[0],split_idx, task, state, {{0},{0}}, &temp1 );
  Contract(&temp1, eff_ham[1], {{0, 2}, {0, 1}}, &temp2);
  temp1 = TenT();
  Contract(&temp2, eff_ham[2],  {{4, 1}, {0, 1}}, &temp1);
  Contract(&temp1, eff_ham[3], {{4, 1}, {1, 0}}, &res);

  //$2
  send_gqten(world, kMasterRank, task, res);//tag = task

  world.recv(kMasterRank, 2*slave_identifier, task);//tag = 2*slave_identifier

  while(task < task_num){
    TenT temp1, temp2, res;
    Contract1Sector(eff_ham[0],split_idx, task, state, {{0},{0}}, &temp1 );
    Contract(&temp1, eff_ham[1], {{0, 2}, {0, 1}}, &temp2);
    temp1 = TenT();
    Contract(&temp2, eff_ham[2],  {{4, 1}, {0, 1}}, &temp1);
    Contract(&temp1, eff_ham[3], {{4, 1}, {1, 0}}, &res);
    send_gqten(world, kMasterRank, task, res);
    world.recv(kMasterRank, 2*slave_identifier, task);
  }


  delete state;
}








}//gqmps2
#endif