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

#include "gqten/gqten.h"
#include "boost/mpi.hpp"


namespace gqmps2{
using namespace gqten;
const size_t kMasterRank = 0;

enum VMPS_ORDER {program_start, lanczos_start, recv_eff_ham, lanczos_mat_vec, lanczos_first_iteration, lanczos_finish,  program_final};


const size_t two_site_eff_ham_size = 4;
namespace mpi = boost::mpi;

/*
inline void MasterSendOrder(const std::string order,
                    const size_t dest,
                    const size_t tag,
                    mpi::communicator world){
    assert(world.rank()==kMasterRank);
    world.send(dest, tag, order);
}
*/

inline void MasterBroadcastOrder(const VMPS_ORDER order,
                mpi::communicator world){
    mpi::broadcast(world, const_cast<VMPS_ORDER&>(order), kMasterRank);
}

inline VMPS_ORDER SlaveGetBroadcastOrder(mpi::communicator world){
  VMPS_ORDER order;
  mpi::broadcast(world, order, kMasterRank);
  return order;
}

template <typename TenElemT, typename QNT>
inline VMPS_ORDER SlaveGetAndDoBroadcastOrder(mpi::communicator world){
VMPS_ORDER order;
mpi::broadcast(world, order, kMasterRank);
using TenT = GQTensor<TenElemT, QNT>;
switch(order){
  case program_start:
    std::cout << "Slave " << world.rank() << "receive program start order" << std::endl;
    break;
  case lanczos_start:
    
    break;
        case recv_eff_ham:{
            std::vector< TenT *> eff_ham(two_site_eff_ham_size);
            for(size_t i=0;i<two_site_eff_ham_size;i++)
                eff_ham[i] = new TenT();
            SlaveRecvEffectiveHamiltonian(eff_ham, world);
           }break;  
        case lanczos_first_iteration:
            break;
        default:
            std::cout << "Slave " << world.rank() << " doesn't understand the order " << order << std::endl;
            break;
    }
    return order;
}
/*
template <typename TenElemT, typename QNT>
void MasterBroadcastEffectiveHamiltonian(
  const std::vector<GQTensor<TenElemT, QNT> *>eff_ham,
  mpi::communicator world
){
  assert(world.rank()==kMasterRank);
  //before this function, broadcast order recv_eff_ham
  using TenT = GQTensor<TenElemT, QNT>;
  for(size_t i=0;i<two_site_eff_ham_size;i++){
    mpi::broadcast(world, (*eff_ham[i]), kMasterRank);
  }
  for(size_t i=0;i<two_site_eff_ham_size;i++){
    const BlockSparseDataTensor<TenElemT, QNT>& bsdt=eff_ham[i]->GetBlkSparDataTen();
    const ElemT* raw_data_pointer = bsdt.GetActualRawDataPtr();
    int raw_data_size = bsdt.GetActualRawDataSize();
    ElemT zero = TenElemT(0.0);
    if( gqten.IsScalar() && data_size==0 ){
        raw_data_pointer = &zero;
        raw_data_size = 1;
    }
    mpi::broadcast(world, raw_data_pointer, raw_data_size , kMasterRank);
  }
}
*/

/*
/// @note new eff_ham[i] outsize
template <typename TenElemT, typename QNT>
void SlaveRecvEffectiveHamiltonian(
    std::vector<GQTensor<TenElemT, QNT> *>eff_ham,
    mpi::communicator world
){
    assert(world.rank()!=kMasterRank);
    assert(eff_ham.size() == two_site_eff_ham_size);
    using TenT = GQTensor<TenElemT, QNT>;
    for(size_t i = 0;i<two_site_eff_ham_size;i++){
        mpi::broadcast(world, (*eff_ham[i]), kMasterRank);
    }
    for(size_t i = 0;i<two_site_eff_ham_size;i++){
        const BlockSparseDataTensor<ElemT, QNT>& bsdt=eff_ham[i]->GetBlkSparDataTen();
        const ElemT* raw_data_pointer = bsdt.GetActualRawDataPtr();
        int raw_data_size = bsdt.GetActualRawDataSize();
        mpi::broadcast(world, raw_data_pointer, raw_data_size , kMasterRank);
    }
    return;
}
*/




}