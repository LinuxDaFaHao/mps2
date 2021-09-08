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
#include "gqmps2/algo_mpi/framework.h"
#include "gqten/gqten.h"

namespace gqmps2 {
using namespace gqten;

namespace mpi = boost::mpi;


//Forward deceleration
template <typename ElemT, typename QNT>
GQTEN_Double master_two_site_eff_ham_mul_state(
    const std::vector<GQTensor<ElemT, QNT> *> &,
    GQTensor<ElemT, QNT> *,
    GQTensor<ElemT, QNT> &,
    mpi::communicator 
);

template <typename ElemT, typename QNT>
void slave_two_site_eff_ham_mul_state(
    const std::vector<GQTensor<ElemT, QNT> *> &,
    mpi::communicator
);




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
    mpi::communicator& world
) {
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  //Broadcast eff_ham, TODO omp parallel
  const size_t eff_ham_size = pinit_state->Rank();//4
#ifdef GQMPS2_TIMING_MODE
  Timer broadcast_eff_ham_timer("broadcast_eff_ham_send");
#endif
  for(size_t i = 0; i < eff_ham_size; i++){
    SendBroadCastGQTensor(world, (*rpeff_ham[i]), kMasterRank);
  }
#ifdef GQMPS2_TIMING_MODE
  broadcast_eff_ham_timer.PrintElapsed();
  Timer lancozs_after_send_ham("lancz_total_time_after_send_ham");
#endif


  LanczosRes<TenT> lancz_res;

  std::vector<TenT *> bases(params.max_iterations);
  std::vector<GQTEN_Double> a(params.max_iterations, 0.0);
  std::vector<GQTEN_Double> b(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
#ifdef GQMPS2_TIMING_MODE
  Timer normalize_timer("lancz_normlize");
#endif
  pinit_state->Normalize();
  bases[0] = pinit_state;

#ifdef GQMPS2_TIMING_MODE
  normalize_timer.PrintElapsed();
  Timer mat_vec_timer("lancz_mat_vec_and_overlap");
#endif
  //first time matrix multiply state will always be done, so here don't need to send order
  TenT* last_mat_mul_vec_res = new TenT();
  a[0] = master_two_site_eff_ham_mul_state(
                                      rpeff_ham, 
                                      bases[0], 
                                      *last_mat_mul_vec_res, 
                                      world
                                      );

#ifdef GQMPS2_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif
  size_t m = 0;
  GQTEN_Double energy0;
  energy0 = a[0];
  // Lanczos iterations.
  while (true) {
    m += 1;
#ifdef GQMPS2_TIMING_MODE
    Timer linear_combine_timer("lancz_linear_combine");
#endif
    auto gamma = last_mat_mul_vec_res;
    if (m == 1) {
      LinearCombine({-a[m-1]}, {bases[m-1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m-1], -b[m-2]},
          {bases[m-1], bases[m-2]},
          1.0,
          gamma
      );
    }
#ifdef GQMPS2_TIMING_MODE
    linear_combine_timer.PrintElapsed();
    normalize_timer.ClearAndRestart();
#endif
    auto norm_gamma = gamma->Normalize();
#ifdef GQMPS2_TIMING_MODE
    normalize_timer.PrintElapsed();
    Timer trigssolver("lancz_triadiag_gs_solver_total");
    trigssolver.Suspend();
#endif
    GQTEN_Double eigval;
    GQTEN_Double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
#ifdef GQMPS2_TIMING_MODE
        Timer lancozs_post_precessing("lancz_post_processing");
#endif
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new TenT(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world);
#ifdef GQMPS2_TIMING_MODE
        lancozs_post_precessing.PrintElapsed();
        lancozs_after_send_ham.PrintElapsed();
#endif
        return lancz_res;
      } else {
#ifdef GQMPS2_TIMING_MODE
        trigssolver.Restart();
#endif
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
#ifdef GQMPS2_TIMING_MODE
        trigssolver.PrintElapsed();
        Timer final_linear_combine_timer("lancz_finial_linear_combine");
#endif
        auto gs_vec = new TenT(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
#ifdef GQMPS2_TIMING_MODE
        final_linear_combine_timer.PrintElapsed();
#endif
#ifdef GQMPS2_TIMING_MODE
        Timer lancozs_post_precessing("lancz_post_processing");
#endif
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world);
#ifdef GQMPS2_TIMING_MODE
        lancozs_post_precessing.PrintElapsed();
        lancozs_after_send_ham.PrintElapsed();
#endif
        return lancz_res;
      }
    }

    b[m-1] = norm_gamma;
    bases[m] = gamma;

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif
    MasterBroadcastOrder(lanczos_mat_vec, world);
    last_mat_mul_vec_res = new TenT();
    a[m] = master_two_site_eff_ham_mul_state(
                                      rpeff_ham, 
                                      bases[m], 
                                      *last_mat_mul_vec_res, 
                                      world
                                      );
#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.PrintElapsed();
    trigssolver.Restart();
#endif

    TridiagGsSolver(a, b, m+1, eigval, eigvec, 'N');
#ifdef GQMPS2_TIMING_MODE
    trigssolver.Suspend();
#endif    
    auto energy0_new = eigval;
    if (
        ((energy0 - energy0_new) < params.error) ||
        (m == eff_ham_eff_dim) ||
        (m == params.max_iterations - 1)
    ) {
#ifdef GQMPS2_TIMING_MODE
      trigssolver.Restart();
#endif
      TridiagGsSolver(a, b, m+1, eigval, eigvec, 'V');
#ifdef GQMPS2_TIMING_MODE
      trigssolver.PrintElapsed();
      Timer final_linear_combine_timer("lancz_finial_linear_combine");
#endif    
      energy0 = energy0_new;
      auto gs_vec = new TenT(bases[0]->GetIndexes());
      LinearCombine(m+1, eigvec, bases, 0.0, gs_vec);
#ifdef GQMPS2_TIMING_MODE
      final_linear_combine_timer.PrintElapsed();
#endif
#ifdef GQMPS2_TIMING_MODE
        Timer lancozs_post_precessing("lancz_post_processing");
#endif
      lancz_res.iters = m;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, last_mat_mul_vec_res);
      MasterBroadcastOrder(lanczos_finish, world);
#ifdef GQMPS2_TIMING_MODE
        lancozs_post_precessing.PrintElapsed();
        lancozs_after_send_ham.PrintElapsed();
#endif
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}

/**
 *
 * @note deceleration the typename TenT when call this function
*/
template <typename TenT>
std::vector<TenT*> SlaveLanczosSolver(
    mpi::communicator world
){

std::vector< TenT *> rpeff_ham(two_site_eff_ham_size);
for(size_t i=0;i<two_site_eff_ham_size;i++){
  rpeff_ham[i] = new TenT();
}
// Receive Hamiltonian
#ifdef GQMPS_MPI_TIMING_MODE
  Timer broadcast_eff_ham_timer("broadcast_eff_ham_recv");
#endif
for(size_t i=0;i<two_site_eff_ham_size;i++){
  RecvBroadCastGQTensor(world, *rpeff_ham[i], kMasterRank);
}
#ifdef GQMPS_MPI_TIMING_MODE
  broadcast_eff_ham_timer.PrintElapsed();
#endif

VMPS_ORDER order=lanczos_mat_vec ;
while(order == lanczos_mat_vec){
  slave_two_site_eff_ham_mul_state(rpeff_ham, world);
  order = SlaveGetBroadcastOrder(world);
}
assert( order==lanczos_finish );

return rpeff_ham;
}


/**
 * two site effective hamiltonian multiplying on state, the dispatch works of master.
 * and calculation the overlap by the way
 * 
 * Once the order lanczos_mat_vec has broadcast to Slave, master will prepare to arrage the tasks.
 * tasks from 0 to slave_num-1 will automatically done by salves at first,
 * other tasks will be sorted from difficulty to easy, to get a more balancing assignment.
 * 
 * @param eff_ham   effective hamiltonian
 * @param state     wave function
 * @param res       the result of effective hamiltonian multiple state,
 *                  as return, need to be a default tensor
 * @param world     boost::mpi::communicator
 * 
 * @return overlap of <res|(*state)>
 */
template <typename ElemT, typename QNT>
GQTEN_Double master_two_site_eff_ham_mul_state(
    const std::vector<GQTensor<ElemT, QNT> *> &eff_ham,
    GQTensor<ElemT, QNT> *state,
    GQTensor<ElemT, QNT> &res,
    mpi::communicator world
) {
  assert(res.IsDefault());
  using TenT = GQTensor<ElemT, QNT>;
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer broadcast_state_timer(" broadcast_state_send");
#endif
  SendBroadCastGQTensor(world, *state, kMasterRank);
#ifdef GQMPS2_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  //prepare
  const size_t split_idx = 2;
  const Index<QNT>& splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();//total task number
  const QNSectorVec<QNT>& split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_num);
  const size_t slave_num = world.size() - 1 ; //total number of slaves
  ///< TODO: support other cases
  assert(task_num > 2*slave_num);
  //$1

  TenT res_shell = TenT( state->GetIndexes() );
  std::vector<size_t> arraging_tasks(task_num);
  std::vector<size_t> task_difficuty(task_num);
  std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_num );
  for(size_t i = 0;i<task_num;i++){
    task_difficuty[i] = split_qnscts[i].GetDegeneracy();
  }

  //I don't know if sort function permits end()<=begin(), although the case almostly cannot occur.
  std::partial_sort(arraging_tasks.begin(), 
                   arraging_tasks.begin()+slave_num,
                   arraging_tasks.end()-slave_num, 
    [&task_difficuty](size_t task1, size_t task2){
      return task_difficuty[task1] > task_difficuty[task2];
       });
  //TODO: add support for multithread, note mpi::environment env( mt::multiple ) outside
  // Also note omp order, maybe need non-block communication
  //if task_num > slave_num:
  //below for loop dispatch tasks from slave_num to task_num-1,
  //from task_num to (slave_num+task_num-1), master informs every slave that the jobs are finished.

  //if task_num <= slave_num, master informs every working slave that the jobs are finished.
  //$2
  // for(size_t task = slave_num; task < slave_num+task_num ; task++){
  for(size_t i = 0; i<task_num;i++){
    if(i == slave_num){ //when arraged the most large slave_num jobs, sort the other work
      std::sort(arraging_tasks.begin()+slave_num, 
                   arraging_tasks.end()-slave_num, 
          [&task_difficuty](size_t task1, size_t task2){
              return task_difficuty[task1] > task_difficuty[task2];
              });
      for(size_t j = i; j<task_num;j++){
        res_list.push_back( res_shell );
      }
    }else{
      res_list.push_back( res_shell );
    }
    // mpi::status recv_status=recv_gqten(world, mpi::any_source, mpi::any_tag, res_list.back());
    auto& bsdt = res_list[i].GetBlkSparDataTen();
    mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
    int slave_identifier = recv_status.source();
    world.send(slave_identifier, 2*slave_identifier, arraging_tasks[i]);
  }
  GQTEN_Double overlap(0.0), overlap_sub;
  for(size_t i = 0; i< std::min(task_num, slave_num) ;i++){
    int slave_identifier = i+1;
    world.recv(slave_identifier, 3*slave_identifier+task_num, overlap_sub);
    overlap += overlap_sub;
  }
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  CollectiveLinearCombine(res_list, res);
#ifdef GQMPS2_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  return overlap;
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
  TenT temp_scalar_ten;
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer broadcast_state_timer(" broadcast_state_recv");
#endif
  RecvBroadCastGQTensor(world, *state, kMasterRank);
#ifdef GQMPS2_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  auto base_dag = Dag(*state);
  // Timer slave_prepare_timer(" slave "+ std::to_string(world.rank()) +"'s prepare");
  const size_t split_idx = 2;
  const Index<QNT>& splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();
  //slave also need to know the total task number used to judge if finish this works
  size_t task_count = 0;
  const size_t slave_identifier = world.rank();//number from 1
  if(slave_identifier > task_num){
    //no task, happy~
#ifdef GQMPS2_MPI_TIMING_MODE
    std::cout << "Slave has done task_count = " << task_count << std::endl;
#endif
    delete state;
    return;
  }
  // slave_prepare_timer.PrintElapsed();
  //$1
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave "+std::to_string(slave_identifier) +"'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave "+ std::to_string(slave_identifier) +"'s work");
#endif
  //first task
  size_t task = slave_identifier-1;
  TenT eff_ham0_times_state;
  TenT temp1, temp2, res;
  //First contract
  TensorContraction1SectorExecutor<ElemT, QNT> ctrct_executor(
    eff_ham[0],
    split_idx,
    task,
    state,
    {{0},{0}},
    &eff_ham0_times_state
  );
  
  ctrct_executor.Execute();

  Contract(&eff_ham0_times_state, eff_ham[1], {{0, 2}, {0, 1}}, &temp2);
  eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
  Contract(&temp2, eff_ham[2],  {{4, 1}, {0, 1}}, &temp1);
  temp2.GetBlkSparDataTen().Clear();
  Contract(&temp1, eff_ham[3], {{4, 1}, {1, 0}}, &res);
  temp1.GetBlkSparDataTen().Clear();

  Contract(
      &res, &base_dag,
      {{0,1,2,3}, {0,1,2,3}},
      &temp_scalar_ten
  );
  GQTEN_Double overlap = Real( temp_scalar_ten() );
  //$2
  // send_gqten(world, kMasterRank, task, res);//tag = task
  auto& bsdt = res.GetBlkSparDataTen();
  // std::cout << "task " << task << " finished, sending " << std::endl;
  task_count++;
#ifdef GQMPS2_MPI_TIMING_MODE
  salve_communication_timer.Restart();
#endif
  bsdt.MPISend(world, kMasterRank, task);//tag = task
  
  world.recv(kMasterRank, 2*slave_identifier, task);//tag = 2*slave_identifier
#ifdef GQMPS2_MPI_TIMING_MODE
  salve_communication_timer.Suspend();
#endif
  while(task < task_num){
    TenT temp1, temp2, res, temp_scalar_ten;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract(&eff_ham0_times_state, eff_ham[1], {{0, 2}, {0, 1}}, &temp2);
    eff_ham0_times_state.GetBlkSparDataTen().Clear();
    Contract(&temp2, eff_ham[2],  {{4, 1}, {0, 1}}, &temp1);
    temp2.GetBlkSparDataTen().Clear();
    Contract(&temp1, eff_ham[3], {{4, 1}, {1, 0}}, &res);
    temp1.GetBlkSparDataTen().Clear();
    
    Contract(
      &res, &base_dag,
      {{0,1,2,3},{0,1,2,3}},
      &temp_scalar_ten
    );
    overlap += Real( temp_scalar_ten() );

    auto& bsdt = res.GetBlkSparDataTen();
    task_count++;
  #ifdef GQMPS2_MPI_TIMING_MODE
    salve_communication_timer.Restart();
  #endif 
    bsdt.MPISend(world, kMasterRank, task);//tag = task
    world.recv(kMasterRank, 2*slave_identifier, task);
  #ifdef GQMPS2_MPI_TIMING_MODE
    salve_communication_timer.Suspend();
  #endif
  }
  world.send(kMasterRank, 3*slave_identifier+task_num, overlap );
#ifdef GQMPS2_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
  std::cout << "Slave " << slave_identifier<< " has done task_count = " << task_count << std::endl;
#endif
  delete state;
}








}//gqmps2
#endif