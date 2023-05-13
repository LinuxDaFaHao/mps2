// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-05-12
*
* Description: GraceQ/MPS2 project. Implementation details for Lanczos solver in DMRG.
*/


#ifndef GQMPS2_ALGO_MPI_LANCZOS_SOLVER_MPI_MASTER_H
#define GQMPS2_ALGO_MPI_LANCZOS_SOLVER_MPI_MASTER_H


/**
@file lanczos_dmrg_solver_mpi_master.h
@brief Implementation details for Lanczos solver in DMRG.
*/
#include "gqmps2/algorithm/lanczos_solver.h"    // LanczosParams
#include "gqten/gqten.h"
#include "gqten/utility/timer.h"                // Timer
#include "gqmps2/algorithm/dmrg/dmrg.h"         // EffectiveHamiltonianTerm
#include "gqmps2/algo_mpi/mps_algo_order.h"

#include <iostream>
#include <vector>     // vector
#include <cstring>

#include "mkl.h"
#include "gqmps2/algo_mpi/dmrg/dmrg_mpi_impl_master.h"

namespace gqmps2 {

using namespace gqten;

/**
Obtain the lowest energy eigenvalue and corresponding eigenstate from the effective
Hamiltonian and a initial state using Lanczos algorithm.

@param rpeff_ham Effective Hamiltonian as a vector of pointer-to-tensors.
@param pinit_state Pointer to initial state for Lanczos iteration.
@param eff_ham_mul_state Function pointer to effective Hamiltonian multiply to state.
@param params Parameters for Lanczos solver.
*/
template<typename TenElemT, typename QNT>
LanczosRes<GQTensor<TenElemT, QNT>> DMRGMPIMasterExecutor<TenElemT,
                                                          QNT>::LanczosSolver_(DMRGMPIMasterExecutor::Tensor *pinit_state) {
  const LanczosParams &params = sweep_params.lancz_params;
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  LanczosRes<Tensor> lancz_res;

  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;
  if (pinit_state->Rank() == 3) {            // For single site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (pinit_state->Rank() == 4) {    // For two site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  }

  std::vector<Tensor *> bases(params.max_iterations, nullptr);
  std::vector<GQTEN_Double> a(params.max_iterations, 0.0);
  std::vector<GQTEN_Double> b(params.max_iterations, 0.0);
  std::vector<GQTEN_Double> N(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
  pinit_state->Normalize();
  bases[0] = pinit_state;

#ifdef GQMPS2_TIMING_MODE
  Timer mat_vec_timer("lancz_mat_vec");
#endif
  MasterBroadcastOrder(lanczos_mat_vec_dynamic, world_);

  auto last_mat_mul_vec_res =
      DynamicHamiltonianMultiplyState_(*bases[0]);

#ifdef GQMPS2_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  Tensor temp_scalar_ten;
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
      LinearCombine({-a[m - 1]}, {bases[m - 1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m - 1], -std::sqrt(N[m - 1])},
          {bases[m - 1], bases[m - 2]},
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
        lancz_res.gs_vec = new Tensor(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world_);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new Tensor(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, m, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world_);
        return lancz_res;
      }
    }

    N[m] = norm_gamma * norm_gamma;
    b[m - 1] = norm_gamma;
    bases[m] = gamma;

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif
    MasterBroadcastOrder(lanczos_mat_vec_static, world_);
    last_mat_mul_vec_res = StaticHamiltonianMultiplyState_(*bases[m], a[m]);

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'N');
    auto energy0_new = eigval;

    if (
        ((energy0 - energy0_new) < params.error) ||
            (m == eff_ham_eff_dim) ||
            (m == params.max_iterations - 1)
        ) {
      TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new Tensor(bases[0]->GetIndexes());
      LinearCombine(m + 1, eigvec, bases, 0.0, gs_vec);
      lancz_res.iters = m + 1;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, m + 1, last_mat_mul_vec_res);
      MasterBroadcastOrder(lanczos_finish, world_);
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}

/*
 * |----1                       1-----
 * |          1        1             |
 * |          |        |             |
 * |          |        |             |
 * |          0        0             |
 * |          1        2             |
 * |          |        |             |
 * |----0 0-------------------3 0----|
 */

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> *DMRGMPIMasterExecutor<TenElemT, QNT>::DynamicHamiltonianMultiplyState_(
    DMRGMPIMasterExecutor::Tensor &state) {
  size_t num_terms = hamiltonian_terms_.size();
  mpi::broadcast(world_, num_terms, kMasterRank);
  SendBroadCastGQTensor(world_, state, kMasterRank);
  if (num_terms <= slave_num_) {
    for (size_t i = 0; i < num_terms; i++) {
      auto &block_site_terms = hamiltonian_terms_[i].first;
      auto &site_block_terms = hamiltonian_terms_[i].second;
      SendBlockSiteHamiltonianTermGroup_(block_site_terms, i + 1);
      SendSiteBlockHamiltonianTermGroup_(site_block_terms, i + 1);
    }

    auto multiplication_res = std::vector<Tensor>(num_terms);
    auto pmultiplication_res = std::vector<Tensor *>(num_terms);
    const std::vector<TenElemT> &coefs = std::vector<TenElemT>(num_terms, TenElemT(1.0));
    for (size_t i = 0; i < num_terms; i++) {
      pmultiplication_res[i] = &multiplication_res[i];
    }

    for (size_t i = 0; i < num_terms; i++) {
      recv_gqten(world_, mpi::any_source, mpi::any_tag, multiplication_res[i]);
    }
    auto res = new Tensor();
    //TODO: optimize the summation
    LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), res);
    return res;
  } else {
    const size_t task_num = num_terms;
    for (size_t task_id = 0; task_id < slave_num_; task_id++) {
      const size_t slave_id = task_id + 1;
      world_.send(slave_id, slave_id, task_id);
      auto &block_site_terms = hamiltonian_terms_[task_id].first;
      auto &site_block_terms = hamiltonian_terms_[task_id].second;
      SendBlockSiteHamiltonianTermGroup_(block_site_terms, slave_id);
      SendSiteBlockHamiltonianTermGroup_(site_block_terms, slave_id);
    }

    auto multiplication_res = std::vector<Tensor>(num_terms);
    auto pmultiplication_res = std::vector<Tensor *>(num_terms);
    const std::vector<TenElemT> &coefs = std::vector<TenElemT>(num_terms, TenElemT(1.0));

    for (size_t i = 0; i < num_terms; i++) {
      pmultiplication_res[i] = &multiplication_res[i];
    }

    for (size_t task_id = slave_num_; task_id < task_num; task_id++) {
      auto recv_status = recv_gqten(world_, mpi::any_source, mpi::any_tag, multiplication_res[task_id - slave_num_]);
      size_t slave_id = recv_status.source();
      world_.send(slave_id, slave_id, task_id);
      auto &block_site_terms = hamiltonian_terms_[task_id].first;
      auto &site_block_terms = hamiltonian_terms_[task_id].second;
      SendBlockSiteHamiltonianTermGroup_(block_site_terms, slave_id);
      SendSiteBlockHamiltonianTermGroup_(site_block_terms, slave_id);
    }

    for (size_t i = task_num - slave_num_; i < task_num; i++) {
      auto recv_status = recv_gqten(world_, mpi::any_source, mpi::any_tag, multiplication_res[i]);
      size_t slave_id = recv_status.source();
      world_.send(slave_id, slave_id, task_num + 10086);//10086 is chosen to make a mock.
    }
    auto res = new Tensor();
    //TODO: optimize the summation
    LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), res);
    return res;
  }
}

template<typename TenElemT, typename QNT>
GQTensor<TenElemT, QNT> *DMRGMPIMasterExecutor<TenElemT, QNT>::StaticHamiltonianMultiplyState_(
    DMRGMPIMasterExecutor::Tensor &state,
    GQTEN_Double &overlap) {
  SendBroadCastGQTensor(world_, state, kMasterRank);
  const size_t num_terms = hamiltonian_terms_.size();
  const size_t gather_terms = std::min(num_terms, slave_num_);
  auto multiplication_res = std::vector<Tensor>(gather_terms);
  auto pmultiplication_res = std::vector<Tensor *>(gather_terms);
  const std::vector<TenElemT> &coefs = std::vector<TenElemT>(gather_terms, TenElemT(1.0));
  for (size_t i = 0; i < gather_terms; i++) {
    pmultiplication_res[i] = &multiplication_res[i];
  }
  for (size_t i = 0; i < gather_terms; i++) {
    recv_gqten(world_, mpi::any_source, mpi::any_tag, multiplication_res[i]);
  }
  auto res = new Tensor();
  LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), res);

  MPI_Barrier(MPI_Comm(world_));
  for (size_t i = 1; i <= gather_terms; i++) {
    GQTEN_Double sub_overlap;
    world_.recv(i, i, sub_overlap);
    overlap += sub_overlap;
  }
  return res;
}



} /* gqmps2 */

#endif