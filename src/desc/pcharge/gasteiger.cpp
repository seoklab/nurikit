//
// Project NuriKit - Copyright 2025 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/log/absl_log.h>
#include <Eigen/Dense>

#include "nuri/eigen_config.h"
#include "nuri/core/graph/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/desc/pcharge.h"

namespace nuri {
namespace {
  template <size_t N>
  class GasteigerParams {
  public:
    constexpr static size_t kNParams = 4;
    constexpr static size_t kNTypes = N;

    constexpr GasteigerParams(const double (&abc)[kNTypes][kNParams - 1])
        : data_ {} {
      for (int i = 1; i < kNTypes; ++i) {
        data_[0][i] = abc[i][0];
        data_[1][i] = abc[i][1];
        data_[2][i] = abc[i][2];
        data_[3][i] = 1 / (abc[i][0] + abc[i][1] + abc[i][2]);
      }

      // Hydrogen has different normalization factor
      data_[3][1] = 1 / 20.02;
    }

    auto array() const {
      // Eigen is col-major
      return Array<double, kNTypes, kNParams>::MapAligned(data_[0]);
    }

    constexpr int ntypes() const { return static_cast<int>(kNTypes); }

  private:
    alignas(Eigen::AlignedMax) double data_[kNParams][kNTypes];
  };

  template <size_t N>
  constexpr GasteigerParams<N>
  make_gasteiger_params(const double (&abc)[N][3]) {
    return GasteigerParams<N>(abc);
  }

  // Order: a, b, c
  constexpr auto kGasteigerParams = make_gasteiger_params({
      {     0,     0,      0 }, // X
      {  7.17,  6.24,  -0.56 }, // H
      {  7.98,  9.18,   1.88 }, // C, sp3
      {  8.79,  9.32,   1.51 }, // C, sp2
      { 10.39,  9.45,   0.73 }, // C, sp
      { 11.54, 10.82,   1.36 }, // N, sp3
      { 12.87, 11.15,   0.85 }, // N, sp2
      { 15.68,  11.7,  -0.27 }, // N, sp
      { 14.18, 12.92,   1.39 }, // O, sp3
      { 17.07, 13.79,   0.47 }, // O, sp2
      { 14.66, 13.85,   2.31 }, // F, sp3
      { 11.00,  9.69,   1.35 }, // Cl, sp3
      { 10.08,  8.47,   1.16 }, // Br, sp3
      {  9.90,  7.96,   0.96 }, // I, sp3
      { 10.14,  9.13,   1.38 }, // S, sp3/so
      { 12.00, 10.81,   1.20 }, // S, so2
      { 10.88,  9.49,   1.33 }, // S, sp2
      {  8.90,  8.24,   0.96 }, // P, sp3
      { 9.665, 8.530,  0.735 }, // P, sp2
      { 7.300, 6.567,  0.657 }, // Si, sp3
      { 7.905, 6.748,  0.443 }, // Si, sp2
      { 9.065, 7.027, -0.002 }, // Si, sp
      { 5.980, 6.820,  1.605 }, // B, sp3
      { 6.420, 6.807,  1.322 }, // B, sp2
      { 3.845, 6.755,  3.165 }, // Be, sp3
      { 4.005, 6.725,  3.035 }, // Be, sp2
      { 3.565, 5.572,  2.197 }, // Mg, sp2
      { 3.300, 5.587,  2.447 }, // Mg, sp3
      { 4.040, 5.472,  1.823 }, // Mg, sp
      { 5.375, 4.953,  0.867 }, // Al, sp3
      { 5.795, 5.020,  0.695 }, // Al, sp2
  });

  using Hyb = constants::Hybridization;

  bool group_16_sp2(const AtomData &data) {
    return data.hybridization() == Hyb::kSP2
           || (data.hybridization() == Hyb::kTerminal
               && (data.formal_charge() == 0
                   || (data.formal_charge() == -1 && data.is_conjugated())));
  }

  bool group_16_sp3(const AtomData &data) {
    return data.hybridization() == Hyb::kSP3
           || (data.hybridization() == Hyb::kTerminal
               && data.formal_charge() == -1 && !data.is_conjugated());
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wswitch-enum"

  int resolve_gasteiger_type(Molecule::Atom atom) {
    auto hyb = atom.data().hybridization();
    if (hyb == Hyb::kUnbound)
      return 0;

    switch (atom.data().atomic_number()) {
    case 0:
      return 0;
    case 1:
      return 1;
    case 6:
      switch (hyb) {
      case Hyb::kSP3:
        return 2;
      case Hyb::kSP2:
        return 3;
      case Hyb::kSP:
      case Hyb::kTerminal:
        return 4;
      }
      break;
    case 7:
      switch (hyb) {
      case Hyb::kSP3:
        return 5;
      case Hyb::kSP2:
        return 6;
      case Hyb::kSP:
      case Hyb::kTerminal:
        return 7;
      }
      break;
    case 8:
      if (group_16_sp3(atom.data()))
        return 8;
      if (group_16_sp2(atom.data()))
        return 9;

      // "looks like" an sp hybridized atom
      if (hyb == Hyb::kTerminal)
        hyb = Hyb::kSP;
      break;
    case 9:
    case 17:
    case 35:
    case 53:
      if (hyb == Hyb::kTerminal)
        return 10 + (atom.data().element().period() - 2);
      break;
    case 16: {
      int num_oxo = absl::c_count_if(atom, [](const Molecule::Neighbor nei) {
        return nei.dst().data().atomic_number() == 8
               && all_neighbors(nei.dst()) == 1;
      });

      if (num_oxo == 1 && hyb == Hyb::kSP2)
        return 14;  // sp3/so
      if (num_oxo > 0 && hyb == Hyb::kSP3)
        return 15;  // "so2"

      if (group_16_sp3(atom.data()))
        return 14;  // sp3/so
      if (group_16_sp2(atom.data()))
        return 16;  // sp2

      // "looks like" an sp hybridized atom
      if (hyb == Hyb::kTerminal)
        hyb = Hyb::kSP;
      break;
    }
    case 15:
      switch (hyb) {
      case Hyb::kSP3:
        return 17;
      case Hyb::kSP2:
        return 18;
      }
      break;
    case 14:
      switch (hyb) {
      case Hyb::kSP3:
        return 19;
      case Hyb::kSP2:
        return 20;
      case Hyb::kSP:
      case Hyb::kTerminal:
        return 21;
      }
      break;
    case 5:
      switch (hyb) {
      case Hyb::kSP3:
        return 22;
      case Hyb::kSP2:
        return 23;
      }
      break;
    case 4:
      switch (hyb) {
      case Hyb::kSP3:
        return 24;
      case Hyb::kSP2:
        return 25;
      }
      break;
    case 12:
      switch (hyb) {
      case Hyb::kSP3:
        return 26;
      case Hyb::kSP2:
        return 27;
      case Hyb::kSP:
      case Hyb::kTerminal:
        return 28;
      }
      break;
    case 13:
      switch (hyb) {
      case Hyb::kSP3:
        return 29;
      case Hyb::kSP2:
        return 30;
      }
      break;
    default:
      break;
    }

    ABSL_LOG(WARNING) << "Unsupported element and hybridization combination: "
                      << atom.data().element_symbol() << " " << hyb;
    return -1;
  }

#pragma GCC diagnostic pop

  bool assign_initial(const Molecule &mol, ArrayXi &atom_types,
                      ArrayX3d &en_pchg_buf) {
    for (auto atom: mol) {
      int atom_type = atom_types[atom.id()] = resolve_gasteiger_type(atom);
      if (atom_type < 0)
        return false;

      en_pchg_buf(atom.id(), 1) = atom.data().formal_charge();
    }

    std::vector<std::vector<int>> type_neighs(kGasteigerParams.ntypes());
    for (auto atom: mol) {
      if (count_heavy(atom) < 2 || atom.data().atomic_number() < 3)
        continue;

      int total_conjugated_neighbors =
          absl::c_count_if(atom, [](const Molecule::Neighbor nei) {
            return nei.edge_data().is_conjugated()
                   && count_heavy(nei.dst()) == 1;
          });
      if (total_conjugated_neighbors < 2)
        continue;

      for (auto &nei: type_neighs)
        nei.clear();

      for (auto nei: atom) {
        if (nei.edge_data().is_conjugated() && count_heavy(nei.dst()) == 1)
          type_neighs[atom_types[nei.dst().id()]].push_back(nei.dst().id());
      }
      if (absl::c_none_of(type_neighs, [](const std::vector<int> &v) {
            return v.size() > 1;
          }))
        continue;

      for (auto &neighs: type_neighs) {
        if (neighs.size() < 2)
          continue;

        en_pchg_buf(neighs, 1) = en_pchg_buf(neighs, 1).mean();
      }
    }

    en_pchg_buf.col(1).tail(mol.size()).setZero();

    return true;
  }
}  // namespace

bool assign_charges_gasteiger(Molecule &mol, int relaxation_steps) {
  const int natoms = mol.size();

  ArrayXi atom_types(natoms);
  ArrayX3d en_pchg_buf(natoms * 2, 3);
  if (!assign_initial(mol, atom_types, en_pchg_buf))
    return false;

  ArrayXd nimpl(natoms);
  for (auto atom: mol)
    nimpl[atom.id()] = static_cast<double>(atom.data().implicit_hydrogens());

  auto en = en_pchg_buf.col(0), pchg = en_pchg_buf.col(1),
       buf = en_pchg_buf.col(2);

  const auto params = kGasteigerParams.array();
  double factor = 1;
  for (int i = 0; i < relaxation_steps; ++i) {
    en.head(natoms) = params(atom_types, 0)
                      + params(atom_types, 1) * pchg.head(natoms)
                      + params(atom_types, 2) * pchg.head(natoms).square();
    en.tail(natoms) = params(1, 0) + params(1, 1) * pchg.tail(natoms)
                      + params(1, 2) * pchg.tail(natoms).square();

    for (auto atom: mol) {
      auto nix = as_index(atom);
      buf[atom.id()] = ((en(nix) >= en[atom.id()])
                            .select(params(atom_types[atom.id()], 3),
                                    params(atom_types(nix), 3))
                        * (en(nix) - en[atom.id()]))
                           .sum();
    }

    buf.tail(natoms) = (en.head(natoms) >= en.tail(natoms))
                           .select(params(1, 3), params(atom_types, 3))
                       * (en.head(natoms) - en.tail(natoms));
    buf.head(natoms) -= nimpl * buf.tail(natoms);

    factor *= 0.5;
    pchg += factor * buf;
  }

  mol.add_prop("mol2_charge_type", "GASTEIGER");
  for (auto atom: mol)
    atom.data().set_partial_charge(pchg[atom.id()]);

  return true;
}
}  // namespace nuri
