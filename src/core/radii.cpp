//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/core/radii.h"

#include <string_view>
#include <utility>

#include <absl/base/attributes.h>
#include <absl/container/flat_hash_map.h>
#include <absl/types/span.h>

#include "nuri/core/container/container_ext.h"
#include "nuri/core/molecule.h"

namespace nuri {
namespace {
  enum class ConnollyKind : int {
    kUnk = 0,
    kC,
    kCH,
    kCH2,
    kCH3,
    kAromaticCH,
    kN,
    kNH2,
    kNH3,
    kO,
    kOH,
    kS,
    kSe,
    kP,
  };

  constexpr double kConnollyRadii[] = {
    kUnknownRadius, 1.80, 1.85, 1.90, 1.95, 1.90, 1.65,
    1.70,           1.75, 1.60, 1.70, 1.90, 1.90, 1.80,
  };

  static_assert(array_size(kConnollyRadii)
                    == static_cast<int>(ConnollyKind::kP) + 1,
                "kConnollyRadii table size mismatch");

  constexpr double connolly_radius(ConnollyKind kind) {
    return kConnollyRadii[static_cast<int>(kind)];
  }

  constexpr ConnollyKind kConnollyCNO[3][4] = {
    { ConnollyKind::kC, ConnollyKind::kCH, ConnollyKind::kCH2,
     ConnollyKind::kCH3 },
    { ConnollyKind::kN,  ConnollyKind::kN, ConnollyKind::kNH2,
     ConnollyKind::kNH3 },
    { ConnollyKind::kO, ConnollyKind::kOH, ConnollyKind::kUnk,
     ConnollyKind::kUnk },
  };

  struct ConnollyAtom {
    std::string_view name;
    ConnollyKind kind;
  };

  constexpr ConnollyAtom kConnollyBackbone[] = {
    {   "N",  ConnollyKind::kN },
    {  "CA", ConnollyKind::kCH },
    {   "C",  ConnollyKind::kC },
    {   "O",  ConnollyKind::kO },
    { "OXT",  ConnollyKind::kO },
  };

  constexpr ConnollyAtom kConnollyAla[] = {
    { "CB", ConnollyKind::kCH3 }
  };
  constexpr ConnollyAtom kConnollyArg[] = {
    {  "CB", ConnollyKind::kCH2 },
    {  "CG", ConnollyKind::kCH2 },
    {  "CD", ConnollyKind::kCH2 },
    {  "NE",   ConnollyKind::kN },
    {  "CZ",   ConnollyKind::kC },
    { "NH1", ConnollyKind::kNH2 },
    { "NH2", ConnollyKind::kNH2 },
  };
  constexpr ConnollyAtom kConnollyAsn[] = {
    {  "CB", ConnollyKind::kCH2 },
    {  "CG",   ConnollyKind::kC },
    { "OD1",   ConnollyKind::kO },
    { "ND2", ConnollyKind::kNH2 },
  };
  constexpr ConnollyAtom kConnollyAsp[] = {
    {  "CB", ConnollyKind::kCH2 },
    {  "CG",   ConnollyKind::kC },
    { "OD1",   ConnollyKind::kO },
    { "OD2",   ConnollyKind::kO },
  };
  constexpr ConnollyAtom kConnollyCys[] = {
    { "CB", ConnollyKind::kCH2 },
    { "SG",   ConnollyKind::kS }
  };
  constexpr ConnollyAtom kConnollyGln[] = {
    {  "CB", ConnollyKind::kCH2 },
    {  "CG", ConnollyKind::kCH2 },
    {  "CD",   ConnollyKind::kC },
    { "OE1",   ConnollyKind::kO },
    { "NE2", ConnollyKind::kNH2 },
  };
  constexpr ConnollyAtom kConnollyGlu[] = {
    {  "CB", ConnollyKind::kCH2 },
    {  "CG", ConnollyKind::kCH2 },
    {  "CD",   ConnollyKind::kC },
    { "OE1",   ConnollyKind::kO },
    { "OE2",   ConnollyKind::kO },
  };
  constexpr ConnollyAtom kConnollyGly[] = {
    { "CA", ConnollyKind::kCH2 }
  };
  constexpr ConnollyAtom kConnollyHis[] = {
    {  "CB",        ConnollyKind::kCH2 },
    {  "CG",          ConnollyKind::kC },
    { "ND1",          ConnollyKind::kN },
    { "CD2", ConnollyKind::kAromaticCH },
    { "CE1", ConnollyKind::kAromaticCH },
    { "NE2",          ConnollyKind::kN },
  };
  constexpr ConnollyAtom kConnollyIle[] = {
    {  "CB",  ConnollyKind::kCH },
    { "CG1", ConnollyKind::kCH2 },
    { "CG2", ConnollyKind::kCH3 },
    { "CD1", ConnollyKind::kCH3 },
  };
  constexpr ConnollyAtom kConnollyLeu[] = {
    {  "CB", ConnollyKind::kCH2 },
    {  "CG",  ConnollyKind::kCH },
    { "CD1", ConnollyKind::kCH3 },
    { "CD2", ConnollyKind::kCH3 },
  };
  constexpr ConnollyAtom kConnollyLys[] = {
    { "CB", ConnollyKind::kCH2 },
    { "CG", ConnollyKind::kCH2 },
    { "CD", ConnollyKind::kCH2 },
    { "CE", ConnollyKind::kCH2 },
    { "NZ", ConnollyKind::kNH3 },
  };
  constexpr ConnollyAtom kConnollyMet[] = {
    { "CB", ConnollyKind::kCH2 },
    { "CG", ConnollyKind::kCH2 },
    { "SD",   ConnollyKind::kS },
    { "CE", ConnollyKind::kCH3 },
  };
  constexpr ConnollyAtom kConnollyMse[] = {
    { "CB", ConnollyKind::kCH2 },
    { "CG", ConnollyKind::kCH2 },
    { "SE",  ConnollyKind::kSe },
    { "CE", ConnollyKind::kCH3 },
  };
  constexpr ConnollyAtom kConnollyPhe[] = {
    {  "CB",        ConnollyKind::kCH2 },
    {  "CG",          ConnollyKind::kC },
    { "CD1", ConnollyKind::kAromaticCH },
    { "CD2", ConnollyKind::kAromaticCH },
    { "CE1", ConnollyKind::kAromaticCH },
    { "CE2", ConnollyKind::kAromaticCH },
    {  "CZ", ConnollyKind::kAromaticCH },
  };
  constexpr ConnollyAtom kConnollyPro[] = {
    { "CB", ConnollyKind::kCH2 },
    { "CG", ConnollyKind::kCH2 },
    { "CD", ConnollyKind::kCH2 },
  };
  constexpr ConnollyAtom kConnollySer[] = {
    { "CB", ConnollyKind::kCH2 },
    { "OG",  ConnollyKind::kOH }
  };
  constexpr ConnollyAtom kConnollyThr[] = {
    {  "CB",  ConnollyKind::kCH },
    { "OG1",  ConnollyKind::kOH },
    { "CG2", ConnollyKind::kCH3 },
  };
  constexpr ConnollyAtom kConnollyTrp[] = {
    {  "CB",        ConnollyKind::kCH2 },
    {  "CG",          ConnollyKind::kC },
    { "CD1", ConnollyKind::kAromaticCH },
    { "CD2",          ConnollyKind::kC },
    { "NE1",          ConnollyKind::kN },
    { "CE2",          ConnollyKind::kC },
    { "CE3", ConnollyKind::kAromaticCH },
    { "CZ2", ConnollyKind::kAromaticCH },
    { "CZ3", ConnollyKind::kAromaticCH },
    { "CH2", ConnollyKind::kAromaticCH },
  };
  constexpr ConnollyAtom kConnollyTyr[] = {
    {  "CB",        ConnollyKind::kCH2 },
    {  "CG",          ConnollyKind::kC },
    { "CD1", ConnollyKind::kAromaticCH },
    { "CD2", ConnollyKind::kAromaticCH },
    { "CE1", ConnollyKind::kAromaticCH },
    { "CE2", ConnollyKind::kAromaticCH },
    {  "CZ",          ConnollyKind::kC },
    {  "OH",         ConnollyKind::kOH },
  };
  constexpr ConnollyAtom kConnollyVal[] = {
    {  "CB",  ConnollyKind::kCH },
    { "CG1", ConnollyKind::kCH3 },
    { "CG2", ConnollyKind::kCH3 },
  };

  struct ConnollyResidue {
    std::string_view name;
    absl::Span<const ConnollyAtom> sidechain;
  };

  constexpr ConnollyResidue kConnollyResidues[] = {
    { "ALA", kConnollyAla },
    { "ARG", kConnollyArg },
    { "ASN", kConnollyAsn },
    { "ASP", kConnollyAsp },
    { "CYS", kConnollyCys },
    { "GLN", kConnollyGln },
    { "GLU", kConnollyGlu },
    { "GLY", kConnollyGly },
    { "HIS", kConnollyHis },
    { "ILE", kConnollyIle },
    { "LEU", kConnollyLeu },
    { "LYS", kConnollyLys },
    { "MET", kConnollyMet },
    { "MSE", kConnollyMse },
    { "PHE", kConnollyPhe },
    { "PRO", kConnollyPro },
    { "SER", kConnollySer },
    { "THR", kConnollyThr },
    { "TRP", kConnollyTrp },
    { "TYR", kConnollyTyr },
    { "VAL", kConnollyVal },
  };

  using ConnollyNameKey = std::pair<std::string_view, std::string_view>;

  const absl::flat_hash_map<ConnollyNameKey, ConnollyKind> &
  connolly_name_map() {
    static const absl::flat_hash_map<ConnollyNameKey, ConnollyKind> map = [] {
      absl::flat_hash_map<ConnollyNameKey, ConnollyKind> m;
      for (const ConnollyResidue &res: kConnollyResidues) {
        for (const ConnollyAtom &atom: kConnollyBackbone)
          m.insert_or_assign({ res.name, atom.name }, atom.kind);
        for (const ConnollyAtom &atom: res.sidechain)
          m.insert_or_assign({ res.name, atom.name }, atom.kind);
      }
      return m;
    }();

    return map;
  }
}  // namespace

double ConnollyRadii::operator()(std::string_view resname,
                                 std::string_view atom_name) const {
  auto it = connolly_name_map().find({ resname, atom_name });
  if (it != connolly_name_map().end())
    return connolly_radius(it->second);

  return kUnknownRadius;
}

double ConnollyRadii::operator()(Molecule::Atom atom) const {
  const int z = atom.data().atomic_number(), n_h = count_hydrogens(atom);

  switch (z) {
  case 6:
    if (atom.data().is_aromatic() && n_h == 1)
      return connolly_radius(ConnollyKind::kAromaticCH);
    ABSL_FALLTHROUGH_INTENDED;
  case 7:
  case 8:
    if (n_h >= 4)
      return kUnknownRadius;
    return connolly_radius(kConnollyCNO[z - 6][n_h]);
  case 15:
    return connolly_radius(ConnollyKind::kP);
  case 16:
    return connolly_radius(ConnollyKind::kS);
  case 34:
    return connolly_radius(ConnollyKind::kSe);
  default:
    return kUnknownRadius;
  }
}
}  // namespace nuri
