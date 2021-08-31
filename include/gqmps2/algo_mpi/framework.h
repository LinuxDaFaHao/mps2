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
  const size_t task_size = splited_index.GetQNSctNum();
  const QNSectorVec<QNT>& split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = world.size() - 1 ;
  IndexVec<QNT> res_indexes(3);
  res_indexes[0] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[3];
  res_indexes[2] = InverseIndex(splited_index);
  TenT res_shell = TenT( res_indexes );
  for(size_t j = 0; j<task_size;j++){
        res_list.push_back( res_shell );
  }
  if(slave_size < task_size){
    std::vector<size_t> task_difficuty(task_size);
    for(size_t i = 0;i<task_size;i++){
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    std::vector<size_t> arraging_tasks(task_size-slave_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size );
  #ifdef GQMPS2_MPI_TIMING_MODE
    Timer sort_timer("grow_env_master_sort_task");
  #endif
    std::sort(arraging_tasks.begin(), 
                   arraging_tasks.end(), 
          [&task_difficuty](size_t task1, size_t task2){
              return task_difficuty[task1] > task_difficuty[task2];
              });
  #ifdef GQMPS2_MPI_TIMING_MODE
    sort_timer.PrintElapsed();
  #endif
    #pragma omp parallel default(none)\
                        shared(task_size, slave_size, res_list, world, arraging_tasks)\
                        num_threads(slave_size)
    {
      size_t controlling_slave = omp_get_thread_num()+1;

      auto& bsdt = res_list[controlling_slave-1].GetBlkSparDataTen();
      const size_t task = controlling_slave-1;
      mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);

      #pragma omp for nowait schedule(dynamic)
      for(size_t i = 0; i < task_size - slave_size; i++){
        world.send(controlling_slave, 2*controlling_slave, arraging_tasks[i]);
        auto& bsdt = res_list[i+slave_size].GetBlkSparDataTen();
        bsdt.MPIRecv(world, controlling_slave, arraging_tasks[i]);
      }
      
      world.send(controlling_slave, 2*controlling_slave, 2*task_size);//finish signal
    }
  }else{//slave_size >= task_size
    #pragma omp parallel default(none)\
                        shared(task_size, res_list, world)\
                        num_threads(task_size)
    {
      size_t controlling_slave = omp_get_thread_num() + 1;
      size_t task = controlling_slave - 1;
      auto& bsdt = res_list[task].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);
      world.send(controlling_slave, 2*controlling_slave, 2*task_size);//finish signal
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
  const size_t task_size = splited_index.GetQNSctNum();
  TenT mps_dag = Dag(mps);
  size_t task_count = 0;
  const size_t slave_identifier = world.rank();//number from 1
  if(slave_identifier > task_size){
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
  while(task < task_size){
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


// Note the return tensor are new in the function
// Suppose threads number >= slave number
template <typename TenElemT, typename QNT>
inline GQTensor<TenElemT, QNT>* MasterGrowRightEnvironment(
  const GQTensor<TenElemT, QNT>& renv,
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
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT>& splited_index = mps.GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  const QNSectorVec<QNT>& split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = world.size() - 1 ;
  IndexVec<QNT> res_indexes(3);
  res_indexes[0] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[0];
  res_indexes[2] = InverseIndex(splited_index);
  TenT res_shell = TenT( res_indexes );
  for(size_t j = 0; j<task_size;j++){
        res_list.push_back( res_shell );
  }
  if(slave_size < task_size){
    std::vector<size_t> task_difficuty(task_size);
    for(size_t i = 0;i<task_size;i++){
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    std::vector<size_t> arraging_tasks(task_size-slave_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size );
  #ifdef GQMPS2_MPI_TIMING_MODE
    Timer sort_timer("grow_env_master_sort_task");
  #endif
    std::sort(arraging_tasks.begin(), 
                   arraging_tasks.end(), 
          [&task_difficuty](size_t task1, size_t task2){
              return task_difficuty[task1] > task_difficuty[task2];
              });
  #ifdef GQMPS2_MPI_TIMING_MODE
    sort_timer.PrintElapsed();
  #endif
    #pragma omp parallel default(none)\
                        shared(task_size, slave_size, res_list, world, arraging_tasks)\
                        num_threads(slave_size)
    {
      size_t controlling_slave = omp_get_thread_num()+1;

      auto& bsdt = res_list[controlling_slave-1].GetBlkSparDataTen();
      const size_t task = controlling_slave-1;
      mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);

      #pragma omp for nowait schedule(dynamic)
      for(size_t i = 0; i < task_size - slave_size; i++){
        world.send(controlling_slave, 2*controlling_slave, arraging_tasks[i]);
        auto& bsdt = res_list[i+slave_size].GetBlkSparDataTen();
        bsdt.MPIRecv(world, controlling_slave, arraging_tasks[i]);
      }
      
      world.send(controlling_slave, 2*controlling_slave, 2*task_size);//finish signal
    }
  }else{//slave_size >= task_size
    #pragma omp parallel default(none)\
                        shared(task_size, res_list, world)\
                        num_threads(task_size)
    {
      size_t controlling_slave = omp_get_thread_num() + 1;
      size_t task = controlling_slave - 1;
      auto& bsdt = res_list[task].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);
      world.send(controlling_slave, 2*controlling_slave, 2*task_size);//finish signal
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
inline void SlaveGrowRightEnvironment(
  const GQTensor<TenElemT, QNT>& renv,
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
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT>& splited_index = mps.GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  TenT mps_dag = Dag(mps);
  size_t task_count = 0;
  const size_t slave_identifier = world.rank();//number from 1
  if(slave_identifier > task_size){
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
    &renv,
    {{2},{0}},
    &env_times_mps
  );
  ctrct_executor.Execute();
  Contract(&env_times_mps, &mpo, {{1,2},{1,3}}, &temp);
  env_times_mps.GetBlkSparDataTen().Clear();
  Contract(&temp, &mps_dag,{{3,1},{1,2}}, &res);
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
  while(task < task_size){
    TenT temp, res;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract(&env_times_mps, &mpo, {{1,2},{1,3}}, &temp);
    env_times_mps.GetBlkSparDataTen().Clear();
    Contract(&temp, &mps_dag,{{3,1},{1,2}}, &res);
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