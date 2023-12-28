/*
* Author: Haoxin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-12-13
*
* Description: GraceQ/MPS2 project. Unittests for FiniteMPO
*/

#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "gqmps2/gqmps2.h"

using namespace gqmps2;
using namespace gqten;

using special_qn::U1QN;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;;

struct TestFiniteMPO : public testing::Test {




};
