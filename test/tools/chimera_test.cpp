//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/tools/chimera.h"

#include <filesystem>
#include <fstream>

#include <Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include <gtest/gtest.h>

#include "nuri/eigen_config.h"
#include "fmt_test_common.h"
#include "test_utils.h"
#include "nuri/fmt/pdb.h"

namespace nuri {
namespace internal {
namespace {
namespace fs = std::filesystem;

Matrix3Xd read_first_calphas(const fs::path &path) {
  std::ifstream ifs(path);
  PDBReader reader(ifs);
  PDBModel model = read_pdb_model(reader.next());

  std::vector<int> calphas;
  for (const PDBResidue &res: model.residues()) {
    for (int i: res.atom_idxs()) {
      const PDBAtom &atom = model.atoms()[i];
      if (atom.name() == "CA") {
        calphas.push_back(i);
        break;
      }
    }
  }

  return model.major_conf()(E::all, calphas);
}

TEST(ChimeraTest, MatchMaker) {
  Matrix3Xd query = read_first_calphas(test_data("fixed_ref1.pdb")),
            templ = read_first_calphas(test_data("fixed_ref2.pdb"));

  Isometry3d ref_xform;
  ref_xform.linear() << -0.14150449, 0.67616094, -0.72303725,  //
      -0.26213529, -0.72990790, -0.63128405,                   //
      -0.95460021, 0.10020405, 0.28053089;
  ref_xform.translation() << 16.25602271, 38.51123346, 17.46343176;

  ArrayXi ref_sel(95);
  ref_sel << 16, 40, 78, 80, 99, 102, 103, 104, 105, 106, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127,
      128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
      143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158,
      159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
      174, 175, 176, 177, 178, 179, 180, 183, 184, 185, 186, 187, 188, 189, 190,
      191, 192, 193, 194, 195, 196;

  auto mm = match_maker(query, templ);
  ASSERT_NEAR(std::sqrt(mm.msd), 0.698, 1e-3);

  ASSERT_EQ(mm.sel.size(), 95);
  absl::c_sort(mm.sel);
  for (int i = 0; i < 95; ++i)
    EXPECT_EQ(mm.sel[i], ref_sel[i]) << i;

  NURI_EXPECT_EIGEN_EQ_TOL(mm.xform.linear(), ref_xform.linear(), 1e-6);
  NURI_EXPECT_EIGEN_EQ_TOL(mm.xform.translation(), ref_xform.translation(),
                           1e-5);

  Matrix3Xd aligned(3, query.cols());
  inplace_transform(aligned, mm.xform, query);
  double msd = (aligned - templ).colwise().squaredNorm().mean();
  EXPECT_NEAR(std::sqrt(msd), 7.472, 1e-3);
}
}  // namespace
}  // namespace internal
}  // namespace nuri
