// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-11
*
* Description: GraceQ/MPS2 project. Utility functions for mpi
*/

/**
@file  utility_mpi.h
@brief  Utility functions for mpi.
*/

#ifndef GQMPS2_ALGO_MPI_FRAMEWORK_H
#define GQMPS2_ALGO_MPI_FRAMEWORK_H

#include "gqten/gqten.h"
#include "boost/mpi.hpp"


namespace gqmps2{
using namespace gqten;
const size_t kMasterRank = 0;

///< variational mps orders send by master
enum VMPS_ORDER {
  program_start,        ///< when vmps start
  lanczos,              ///< when lanczos start
  svd,                  ///< before svd
  lanczos_mat_vec,      ///< before do lanczos' matrix vector multiplication
  lanczos_first_iteration,  ///< no use up to now
  lanczos_finish,       ///< when lanczos finished
  growing_left_env,     ///< growing left environment
  growing_right_env,    ///< growing right environment
  program_final         /// when vmps finished
};


const size_t two_site_eff_ham_size = 4;
namespace mpi = boost::mpi;

inline void MasterBroadcastOrder(const VMPS_ORDER order,
                mpi::communicator& world){
    mpi::broadcast(world, const_cast<VMPS_ORDER&>(order), kMasterRank);
}

inline VMPS_ORDER SlaveGetBroadcastOrder(mpi::communicator world){
  VMPS_ORDER order;
  mpi::broadcast(world, order, kMasterRank);
  return order;
}


//Growing environment tensor functions

// Note the return tensor are new in the function
// Suppose threads number >= slave number
template <typename TenElemT, typename QNT>
inline GQTensor<TenElemT, QNT>* MasterGrowLeftEnvironment(
  const GQTensor<TenElemT, QNT>& lenv,
  const GQTensor<TenElemT, QNT>& mpo,
  const GQTensor<TenElemT, QNT>& mps,
  mpi::communicator& world
){
  using TenT = GQTensor<TenElemT, QNT>;
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_send");
#endif
  SendBroadCastGQTensor(world, mps, kMasterRank);
#ifdef GQMPS2_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 2; //index of mps tensor
  const Index<QNT>& splited_index = mps.GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();
  const QNSectorVec<QNT>& split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_num);
  const size_t slave_num = world.size() - 1 ;
  IndexVec<QNT> res_indexes(3);
  res_indexes[0] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[3];
  res_indexes[2] = InverseIndex(splited_index);
  TenT res_shell = TenT( res_indexes );
  std::vector<size_t> task_difficuty(task_num);
  for(size_t i = 0;i<task_num;i++){
    task_difficuty[i] = split_qnscts[i].GetDegeneracy();
  }
  for(size_t j = 0; j<task_num;j++){
        res_list.push_back( res_shell );
  }
  if(slave_num < task_num){
    std::vector<size_t> arraging_tasks(task_num-slave_num);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_num );
  /// TODO: sort timer
    std::sort(arraging_tasks.begin(), 
                   arraging_tasks.end(), 
          [&task_difficuty](size_t task1, size_t task2){
              return task_difficuty[task1] > task_difficuty[task2];
              });
    
    #pragma omp parallel default(none)\
                        shared(task_num, slave_num, res_list, world, arraging_tasks)\
                        num_threads(slave_num)
    {
      #pragma omp for schedule(static)
      for(size_t i = 0; i < slave_num; i++){
        size_t controlling_slave = omp_get_thread_num()+1; //does this line can move out of for loop?
        auto& bsdt = res_list[controlling_slave-1].GetBlkSparDataTen();
        const size_t task = controlling_slave-1;
        mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);
      }
      /// TODO: omp no wait
      #pragma omp for schedule(dynamic)
      for(size_t i = 0; i < task_num - slave_num; i++){
        size_t controlling_slave = omp_get_thread_num()+1;
        world.send(controlling_slave, 2*controlling_slave, arraging_tasks[i]);
        auto& bsdt = res_list[i+slave_num].GetBlkSparDataTen();
        bsdt.MPIRecv(world, controlling_slave, arraging_tasks[i]);
      }
      //omp no wait?
      #pragma omp for schedule(static)
      for(size_t i = 0; i < slave_num; i++){
        size_t controlling_slave = omp_get_thread_num()+1;
        world.send(controlling_slave, 2*controlling_slave, 2*task_num);//finish signal
      }
    }
  }else{//slave_num >= task_num
    #pragma omp parallel for default(none)\
                        shared(task_num, res_list, world)\
                        num_threads(task_num)\
                        schedule(static)
    for(size_t i = 0; i < task_num; i++){
      auto& bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, i+1, i);
    }
    #pragma omp parallel for default(none)\
                        shared(task_num, res_list, world)\
                        num_threads(task_num)\
                        schedule(static)
    for(size_t i = 0; i < task_num; i++){
        world.send(i+1, 2*(i+1), 2*task_num);//finish signal
    }
  }
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  TenT* res = new TenT();
  CollectiveLinearCombine(res_list, *res);
#ifdef GQMPS2_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  return res;
}

template <typename TenElemT, typename QNT>
inline void SlaveGrowLeftEnvironment(
  const GQTensor<TenElemT, QNT>& lenv,
  const GQTensor<TenElemT, QNT>& mpo,
  mpi::communicator& world
){
  using TenT = GQTensor<TenElemT, QNT>;
  TenT mps;
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_recv");
#endif
  RecvBroadCastGQTensor(world, mps, kMasterRank);
#ifdef GQMPS2_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 2; //index of mps tensor
  const Index<QNT>& splited_index = mps.GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();
  TenT mps_dag = Dag(mps);
  size_t task_count = 0;
  const size_t slave_identifier = world.rank();//number from 1
  if(slave_identifier > task_num){
    //no task, happy~
    std::cout << "Slave has done task_count = " << task_count << std::endl;
    return;
  }
#ifdef GQMPS2_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave "+std::to_string(slave_identifier) +"'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave "+ std::to_string(slave_identifier) +"'s work");
#endif
  //first task
  size_t task = slave_identifier-1;
  TenT env_times_mps;
  TenT temp, res;
  //First contract
  TensorContraction1SectorExecutor<TenElemT, QNT> ctrct_executor(
    &mps,
    split_idx,
    task,
    &lenv,
    {{0},{0}},
    &env_times_mps
  );
  ctrct_executor.Execute();
  Contract(&env_times_mps, &mpo, {{2,0},{0,1}}, &temp);
  env_times_mps.GetBlkSparDataTen().Clear();
  Contract(&temp, &mps_dag,{{1,2},{0,1}}, &res);
  temp.GetBlkSparDataTen().Clear();
  auto& bsdt = res.GetBlkSparDataTen();
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
    TenT temp, res;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract(&env_times_mps, &mpo, {{2,0},{0,1}}, &temp);
    env_times_mps.GetBlkSparDataTen().Clear();
    Contract(&temp, &mps_dag,{{1,2},{0,1}}, &res);
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
#ifdef GQMPS2_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
#endif
  std::cout << "Slave " << slave_identifier<< " has done task_count = " << task_count << std::endl;
}


}//gqmps2

#endif