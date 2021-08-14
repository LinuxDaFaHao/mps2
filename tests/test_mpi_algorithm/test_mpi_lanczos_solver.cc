// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-12
*
* Description: GraceQ/mps2 project. MPI Lanczos algorithm unittests.
*/

#include "gqmps2/algorithm/lanczos_solver.h"
#include "../testing_utils.h"
#include "gqten/gqten.h"
#include "gqten/utility/timer.h"
#include "gqmps2/algo_mpi/lanczos_solver_mpi.h"
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


using U1U1QN = QN<U1QNVal, U1QNVal>;
using IndexT = Index<U1U1QN>;
using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1U1QN>;


// TEST(MPI_LANCZOS_TEST, MatrixMultiplyVector){
int main(){
    namespace mpi = boost::mpi;
    using std::vector;
    mpi::environment env;
    mpi::communicator world;
    if( world.rank() == 0){
        hp_numeric::SetTensorManipulationTotalThreads(14);
        hp_numeric::SetTensorTransposeNumThreads(14);
        DGQTensor lenv, renv, mpo1, mpo2, mps1, mps2;
        vector<DGQTensor *> load_ten_list = {&lenv, &renv, &mpo1, &mpo2, &mps1, &mps2};
        vector<std::string > file_name_list = {"lenv.gqten", "renv.gqten", "mpo_ten_l.gqten",
                                "mpo_ten_r.gqten", "mps_ten_l.gqten", "mps_ten_r.gqten"};
        assert(load_ten_list.size() == file_name_list.size());
        for(size_t i =0;i<load_ten_list.size();i++){
            std::string file = file_name_list[i];
            std::ifstream ifs(file, std::ios::binary);
            if(!ifs.good()){
                std::cout << "can not open the file " << file << std::endl;
                exit(1);
            }
            ifs >> *load_ten_list[i];
        }
        std::cout << "Master has loaded the tensors." <<std::endl;

        SendBroadCastGQTensor(world,lenv, kMasterRank);
        SendBroadCastGQTensor(world,renv, kMasterRank);
        SendBroadCastGQTensor(world,mpo1, kMasterRank);
        SendBroadCastGQTensor(world,mpo2, kMasterRank);

        vector<DGQTensor*> eff_ham = {&lenv, &mpo1, &mpo2, &renv};

        DGQTensor* state = new DGQTensor();
        Contract(&mps1, &mps2, {{2},{0}}, state);
        state->ConciseShow();
        Timer mpi_mat_vec_timer("mpi matrix multiply vector");
        DGQTensor* mpi_res = master_two_site_eff_ham_mul_state(eff_ham, state,world);
        mpi_mat_vec_timer.PrintElapsed();
        
        Timer single_process_mat_vec_timer("single process matrix multiply vector");
        DGQTensor* single_process_res = eff_ham_mul_two_site_state(eff_ham, state);
        single_process_mat_vec_timer.PrintElapsed();

        DGQTensor diff =  (*mpi_res) +(-(*single_process_res));
        EXPECT_NEAR(diff.Normalize(), 0.0, 1e-13);
        
    }else{
        hp_numeric::SetTensorManipulationTotalThreads(14);
        hp_numeric::SetTensorTransposeNumThreads(14);
        DGQTensor lenv, renv, mpo1, mpo2, mps1, mps2;
        RecvBroadCastGQTensor(world, lenv, kMasterRank);
        RecvBroadCastGQTensor(world,renv, kMasterRank);
        RecvBroadCastGQTensor(world,mpo1, kMasterRank);
        RecvBroadCastGQTensor(world,mpo2, kMasterRank);
        std::cout << "Slave has received the eff_hams" <<std::endl;
        vector<DGQTensor*> eff_ham = {&lenv, &mpo1, &mpo2, &renv};

        slave_two_site_eff_ham_mul_state(eff_ham, world);
    }
    
    return 0;
}
