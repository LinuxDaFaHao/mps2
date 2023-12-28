// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-12
*
* Description: GraceQ/mps2 project. MPI Lanczos algorithm unittests.
*/

#include "gqmps2/algorithm/lanczos_params.h"
#include "../testing_utils.h"
#include "gqten/gqten.h"
#include "gqten/utility/timer.h"
#include "gqmps2/algo_mpi/vmps/lanczos_solver_mpi_master.h"
#include "boost/mpi.hpp"

#include "gtest/gtest.h"

#include <vector>
#include <iostream>
#include <fstream>

#ifdef Release
#define NDEBUG
#endif

#include <assert.h>

#include "mkl.h"

using namespace gqmps2;
using namespace gqten;
namespace mpi = boost::mpi;

using U1U1QN = QN<U1QNVal, U1QNVal>;
using IndexT = Index<U1U1QN>;
using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1U1QN>;

// TEST(MPI_LANCZOS_TEST, MatrixMultiplyVector){
int main(int argc, char *argv[]) {
  using std::vector;
  mpi::environment env;
  mpi::communicator world;
  size_t thread_num;
  if (argc == 1) {// no input paramter
    thread_num = 12;
  } else {
    thread_num = atoi(argv[1]);
  }
  hp_numeric::SetTensorManipulationThreads(thread_num);
  hp_numeric::SetTensorTransposeNumThreads(thread_num);
  DGQTensor lenv, renv, mpo1, mpo2, mps1, mps2;
  std::vector<DGQTensor *> eff_ham = {&lenv, &mpo1, &mpo2, &renv};
  if (world.rank() == 0) {
    vector<DGQTensor *> load_ten_list = {&lenv, &renv, &mpo1, &mpo2, &mps1, &mps2};
    vector<std::string> file_name_list = {"lenv.gqten", "renv.gqten", "mpo_ten_l.gqten",
                                          "mpo_ten_r.gqten", "mps_ten_l.gqten", "mps_ten_r.gqten"};
    assert(load_ten_list.size() == file_name_list.size());
    Timer load_tensor_timer("load tensors");
    for (size_t i = 0; i < load_ten_list.size(); i++) {
      std::string file = file_name_list[i];
      std::ifstream ifs(file, std::ios::binary);
      if (!ifs.good()) {
        std::cout << "can not open the file " << file << std::endl;
        exit(1);
      }
      ifs >> *load_ten_list[i];
    }
    load_tensor_timer.PrintElapsed();
    std::cout << "Master has loaded the tensors." << std::endl;
  }
  if (world.rank() == 0) {
    DGQTensor *initial_state = new DGQTensor();
    Contract(&mps1, &mps2, {{2}, {0}}, initial_state);
    initial_state->ConciseShow();
    LanczosParams params(1e-10, 5);
    Timer mpi_lanczos_timer("mpi lanczos solver");
    auto lanczos_res = MasterLanczosSolver(eff_ham, initial_state, params, world);
    mpi_lanczos_timer.PrintElapsed();
    std::cout << "MPI lanczos iter = " << lanczos_res.iters
              << "E0 = " << lanczos_res.gs_eng;
    initial_state = new DGQTensor();
    Contract(&mps1, &mps2, {{2}, {0}}, initial_state);
    Timer single_process_lanczos_timer("single processor lanczos solver");
    auto lanczos_res2 = LanczosSolver(eff_ham, initial_state, eff_ham_mul_two_site_state, params);
    single_process_lanczos_timer.PrintElapsed();
    std::cout << "Single processor lanczos iter = " << lanczos_res2.iters
              << "E0 = " << lanczos_res2.gs_eng;
    DGQTensor diff = (*lanczos_res.gs_vec) + (-(*lanczos_res2.gs_vec));
    EXPECT_NEAR((diff.Normalize()) / (lanczos_res2.gs_vec->Normalize()), 0.0, 1e-13);
    delete lanczos_res.gs_vec;
    delete lanczos_res2.gs_vec;
  } else {
    std::vector<DGQTensor *> eff_ham = SlaveLanczosSolver<DGQTensor>(world);
    for (size_t i = 0; i < two_site_eff_ham_size; i++) {
      delete eff_ham[i];
    }
  }
  return 0;
}
