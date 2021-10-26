// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-26
*
* Description: GraceQ/mps2 project. Unittest for MPI two sites algorithm.
*/

#include "gqmps2/gqmps2.h"
#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "boost/mpi.hpp"


#include <vector>

#include <stdlib.h>     // system


using namespace gqmps2;
using namespace gqten;

using U1QN = QN<U1QNVal>;
using U1U1QN = QN<U1QNVal, U1QNVal>;

using IndexT = Index<U1QN>;
using IndexT2 = Index<U1U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctT2 = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;
using QNSctVecT2 = QNSectorVec<U1U1QN>;
using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using DGQTensor2 = GQTensor<GQTEN_Double, U1U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;
using ZGQTensor2 = GQTensor<GQTEN_Complex, U1U1QN>;
using DSiteVec = SiteVec<GQTEN_Double, U1QN>;
using DSiteVec2 = SiteVec<GQTEN_Double, U1U1QN>;
using ZSiteVec = SiteVec<GQTEN_Complex, U1QN>;
using ZSiteVec2 = SiteVec<GQTEN_Complex, U1U1QN>;
using DMPS = FiniteMPS<GQTEN_Double, U1QN>;
using DMPS2 = FiniteMPS<GQTEN_Double, U1U1QN>;
using ZMPS = FiniteMPS<GQTEN_Complex, U1QN>;
using ZMPS2 = FiniteMPS<GQTEN_Complex, U1U1QN>;


// Helpers
inline void KeepOrder(size_t &x, size_t &y) {
  if (x > y) {
    auto temp = y;
    y = x;
    x = temp;
  }
}


inline size_t coors2idx(
    const size_t x, const size_t y, const size_t Nx, const size_t Ny) {
	return x * Ny + y;
}



inline size_t coors2idxSquare(
    const int x, const int y, const size_t Nx, const size_t Ny) {
  return x * Ny + y;
}


inline size_t coors2idxHoneycomb(
    const int x, const int y, const size_t Nx, const size_t Ny) {
  return Ny * (x%Nx) + y%Ny;
}


inline void RemoveFolder(const std::string &folder_path) {
  std::string command = "rm -rf " + folder_path;
  system(command.c_str());
}


// Test spin systems
// struct Test2DSpinSystem : public testing::Test {
  size_t Lx = 4;
  size_t Ly = 4;
  size_t N = Lx*Ly;

  U1QN qn0 = U1QN({QNCard("Sz", U1QNVal(0))});
  IndexT pb_out = IndexT({
                      QNSctT(U1QN({QNCard("Sz", U1QNVal( 1))}), 1),
                      QNSctT(U1QN({QNCard("Sz", U1QNVal(-1))}), 1)},
                      GQTenIndexDirType::OUT
                  );
  IndexT pb_in = InverseIndex(pb_out);
  DSiteVec dsite_vec_2d = DSiteVec(N, pb_out);
  ZSiteVec zsite_vec_2d = ZSiteVec(N, pb_out);

  DGQTensor  did  = DGQTensor({pb_in, pb_out});
  DGQTensor  dsz  = DGQTensor({pb_in, pb_out});
  DGQTensor  dsp  = DGQTensor({pb_in, pb_out});
  DGQTensor  dsm  = DGQTensor({pb_in, pb_out});
  DMPS dmps = DMPS(dsite_vec_2d);

  ZGQTensor  zid  = ZGQTensor({pb_in, pb_out});
  ZGQTensor  zsz  = ZGQTensor({pb_in, pb_out});
  ZGQTensor  zsp  = ZGQTensor({pb_in, pb_out});
  ZGQTensor  zsm  = ZGQTensor({pb_in, pb_out});
  ZMPS zmps = ZMPS(zsite_vec_2d);

  std::vector<std::pair<size_t, size_t>> nn_pairs = 
    std::vector<std::pair<size_t, size_t>>(size_t(2*Lx*Ly-Ly));
  void SetUp(void) {
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;

    auto iter = nn_pairs.begin();
    for(size_t i = 0; i< Lx;i++){
      for(size_t j = 0; j < Ly; j++){
        size_t site_a = i*Ly+j;
        if(j!=Ly-1){
          size_t site_b = site_a+1;
          iter->first = site_a;
          iter->second = site_b;
        }else{
          size_t site_b = i*Ly;
          iter->first = site_b;
          iter->second = site_a;
        }
        iter++;
      }
    }
    for(size_t i = 0; i< Lx-1;i++){
      for(size_t j = 0; j < Ly; j++){
        size_t site_a = i*Ly+j;
        size_t site_b = (i+1)*Ly+j;
        iter->first = site_a;
        iter->second = site_b;
        iter++;
      }
    }
    assert(iter == nn_pairs.end());

  }
// };



// TEST_F(Test2DSpinSystem, 2DHeisenberg) {
int main(){
  namespace mpi = boost::mpi;
  mpi::environment env(mpi::threading::multiple);
  mpi::communicator world;
  SetUp();
  auto dmpo_gen = MPOGenerator<GQTEN_Double, U1QN>(dsite_vec_2d, qn0);

  for (auto &p : nn_pairs) {
    dmpo_gen.AddTerm(1,   {dsz, dsz}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {p.first, p.second});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params1 = SweepParams(
                          4,
                          100, 100, 1.0E-7,
                          LanczosParams(1.0E-7)
                          );

  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs);

  if(world.rank() == 0){
    dmps.Dump(sweep_params1.mps_path, true); 
    if (IsPathExist(sweep_params1.temp_path)){
      RemoveFolder(sweep_params1.temp_path);
    }
    TwoSiteFiniteVMPS(dmps, dmpo, sweep_params1);
  }


  auto sweep_params = TwoSiteMPIVMPSSweepParams(
                          4,
                          100, 100, 1.0E-9,
                          LanczosParams(1.0E-7)
                          );

  double e0 = TwoSiteFiniteVMPS(dmps, dmpo,sweep_params, world);
  
  
  if(world.rank() == 0 ){
    std::cout << "e0 = " << e0 << std::endl;
    EXPECT_NEAR(e0, -10.264281906484872, 1e-5);
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
  }

  //Complex case
  auto zmpo_gen = MPOGenerator<GQTEN_Complex, U1QN>(zsite_vec_2d, qn0);

  for (auto &p : nn_pairs) {
    zmpo_gen.AddTerm(1,   {zsz, zsz}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {p.first, p.second});
  }
  auto zmpo = zmpo_gen.Gen();

  DirectStateInitMps(zmps, stat_labs);
  if(world.rank() == 0){
    zmps.Dump(sweep_params1.mps_path, true);
    if (IsPathExist(sweep_params1.temp_path)){
      RemoveFolder(sweep_params1.temp_path);
    }
    TwoSiteFiniteVMPS(zmps, zmpo, sweep_params1);
  }


  e0 = TwoSiteFiniteVMPS(zmps, zmpo,sweep_params, world);


  if(world.rank() == 0 ){
    std::cout << "e0 = " << e0 << std::endl;
    EXPECT_NEAR(e0, -10.264281906484872, 1e-5);
    RemoveFolder(sweep_params.mps_path);
    RemoveFolder(sweep_params.temp_path);
  }
  return 0;
}




