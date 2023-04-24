// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-21
*
* Description: GraceQ/MPS2 project. Matrix Represented matrix product operator (MPO).
*/


#ifndef GQMPS2_ONE_DIM_TN_MAT_REPR_MPO_H
#define GQMPS2_ONE_DIM_TN_MAT_REPR_MPO_H

#include "gqten/gqten.h"                                        //GQTensor
#include "gqmps2/one_dim_tn/mpo/mpogen/symb_alg/sparse_mat.h"   //SparMat

namespace gqmps2 {
using namespace gqten;

template <typename TenT>
using MatReprMPO = std::vector<SparMat<TenT>>;
}/* gqmps2 */




#endif //GQMPS2_ONE_DIM_TN_MAT_REPR_MPO_H
