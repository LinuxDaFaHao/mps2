// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-22
*
* Description: GraceQ/MPS2 project. Generate the file names.
*/

#ifndef GQMPS2_ALGORITHM_DMRG_OPERATOR_IO_H
#define GQMPS2_ALGORITHM_DMRG_OPERATOR_IO_H

#include "gqten/gqten.h"
#include "gqmps2/consts.h"                //kOpFileBaseName
#include "gqmps2/algorithm/dmrg/dmrg.h"   //RightOperatorGroup
#include "gqmps2/utilities.h"             //WriteGQTensorTOFile

namespace gqmps2 {
using namespace gqten;

inline std::string GenOpFileName(
    const std::string &dir,
    const size_t blk_len,
    const size_t component,
    const std::string &temp_path
) {
  return temp_path + "/" + dir
      + kOpFileBaseName + std::to_string(blk_len)
      + "Comp" + std::to_string(component)
      + "." + kGQTenFileSuffix;
}

template<typename TenT>
void WriteOperatorGroup(
    const std::string &dir,
    const size_t blk_len,
    OperatorGroup<TenT> &op_gp,
    const std::string &temp_path
) {
  for (size_t comp = 0; comp < op_gp.size(); comp++) {
    std::string file_name = GenOpFileName(dir, blk_len, comp, temp_path);
    WriteGQTensorTOFile(op_gp[comp], file_name);
  }
}


///< elements in op_gp are assumed as TenT(), with correct number of elements.
template<typename TenT>
bool ReadOperatorGroup(
    const std::string &dir,
    const size_t blk_len,
    OperatorGroup<TenT> &op_gp,
    const std::string &temp_path
) {
  bool read_success(true);
  for (size_t comp = 0; comp < op_gp.size(); comp++) {
    std::string file_name = GenOpFileName(dir, blk_len, comp, temp_path);
    read_success &= ReadGQTensorFromFile(op_gp[comp], file_name);
  }
  return read_success;
}

template<typename TenT>
bool ReadAndRemoveOperatorGroup(
    const std::string &dir,
    const size_t blk_len,
    OperatorGroup<TenT> &op_gp,
    const std::string &temp_path
) {
  bool read_success(true);
  for (size_t comp = 0; comp < op_gp.size(); comp++) {
    std::string file_name = GenOpFileName(dir, blk_len, comp, temp_path);
    read_success &= ReadGQTensorFromFile(op_gp[comp], file_name);
    RemoveFile(file_name);
  }
  return read_success;
}

}

#endif //GQMPS2_ALGORITHM_DMRG_OPERATOR_IO_H
