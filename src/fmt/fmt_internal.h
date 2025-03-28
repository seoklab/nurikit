//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_FMT_INTERNAL_H_
#define NURI_FMT_FMT_INTERNAL_H_

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/strings/str_cat.h>
#include <boost/container/container_fwd.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/spirit/home/x3.hpp>

#include "nuri/eigen_config.h"
#include "nuri/core/graph.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
template <auto AltlocExtractor, class AT, class CoordConverter>
void pdb_update_confs(Molecule &mol, const std::vector<AT> &atom_data,
                      const CoordConverter &coords) {
  boost::container::flat_set<std::string_view> altlocs;

  for (const AT &ad: atom_data) {
    if (ad.data().size() <= 1)
      continue;

    altlocs.insert(make_transform_iterator<AltlocExtractor>(ad.data().begin()),
                   make_transform_iterator<AltlocExtractor>(ad.data().end()));
  }

  if (altlocs.empty()) {
    Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd(3, atom_data.size()));
    for (int i = 0; i < atom_data.size(); ++i)
      coords(atom_data[i].first(), conf.col(i));
    return;
  }

  ABSL_DCHECK_GT(altlocs.size(), 1);

  mol.confs().resize(altlocs.size());
  for (int i = 0; i < altlocs.size(); ++i)
    mol.confs()[i].resize(3, static_cast<int>(atom_data.size()));

  for (int i = 0; i < atom_data.size(); ++i) {
    const AT &ad = atom_data[i];

    if (ad.data().size() == 1) {
      Eigen::Ref<Vector3d> coord = mol.confs()[0].col(i);
      coords(ad.first(), coord);
      for (int j = 1; j < mol.confs().size(); ++j)
        mol.confs()[j].col(i) = coord;
      continue;
    }

    if (ad.data().size() == altlocs.size()) {
      for (int j = 0; j < mol.confs().size(); ++j)
        coords(ad.data()[j], mol.confs()[j].col(i));
      continue;
    }

    Vector3d major_coord;
    coords(ad.data()[ad.major()], major_coord);

    int j = 0;
    for (int k = 0; k < ad.data().size(); ++j) {
      ABSL_DCHECK(j < mol.confs().size());

      std::string_view altloc = altlocs.begin()[j];
      Eigen::Ref<Vector3d> coord = mol.confs()[j].col(i);

      if (std::invoke(AltlocExtractor, ad.data()[k]) != altloc) {
        coord = major_coord;
        continue;
      }

      coords(ad.data()[k++], coord);
    }

    for (; j < mol.confs().size(); ++j)
      mol.confs()[j].col(i) = major_coord;
  }
}

template <class RT, class AT, class ChainAsSv, class ResidueMember,
          class SeqMember, class ICodeAsSv, class EntityMember = std::nullptr_t,
          std::enable_if_t<!std::is_reference_v<RT>, int> = 0>
void pdb_update_substructs(
    // NOLINTNEXTLINE(*-missing-std-forward)
    Molecule &mol, RT &&residues, const std::vector<AT> &atoms,
    const ChainAsSv &chain_sv, const ResidueMember &residue_member,
    const SeqMember &seq_member, const ICodeAsSv &icode_sv,
    const EntityMember &eid_member = nullptr) {
  auto &subs = mol.substructures();

  boost::container::flat_map<std::string_view, std::vector<int>> chains;
  for (int i = 0; i < atoms.size(); ++i) {
    const auto &id = atoms[i].first().id().res;

    auto [cit, _] = chains.insert({ std::invoke(chain_sv, id), {} });
    cit->second.push_back(i);
  }

  subs.reserve(residues.size() + chains.size());
  subs.resize(residues.size(), mol.substructure(SubstructCategory::kResidue));
  subs.resize(residues.size() + chains.size(),
              mol.substructure(SubstructCategory::kChain));

  auto sit = subs.begin();
  for (int i = 0; i < residues.size(); ++i, ++sit) {
    auto &data = residues[i];

    sit->update(IndexSet(boost::container::ordered_unique_range,
                         std::move(data.idxs)),
                {});
    sit->name() = std::invoke(residue_member, data);
    sit->set_id(std::invoke(seq_member, data.id));
    // Workaround GCC bug; produces "basic_string::_S_construct null not valid"
    // exception when called directly with type std::string_view &
    sit->add_prop("chain", std::string { std::invoke(chain_sv, data.id) });

    std::string_view icode = std::invoke(icode_sv, data.id);
    if (!icode.empty())
      sit->add_prop("icode", icode);

    if constexpr (!std::is_same_v<EntityMember, std::nullptr_t>) {
      sit->add_prop("entity_id",
                    std::invoke(eid_member, atoms[sit->atom_ids()[0]]));
    }
  }

  for (int i = 0; i < chains.size(); ++i, ++sit) {
    auto &chain = chains.begin()[i];
    sit->update(IndexSet(boost::container::ordered_unique_range,
                         std::move(chain.second)),
                {});
    sit->name() = chain.first;
    sit->set_id(i);

    if constexpr (!std::is_same_v<EntityMember, std::nullptr_t>) {
      sit->add_prop("entity_id",
                    std::invoke(eid_member, atoms[sit->atom_ids()[0]]));
    }
  }
}
}  // namespace internal

// NOLINTNEXTLINE(google-build-namespaces)
namespace {
// NOLINTBEGIN(readability-identifier-naming,*-unused-const-variable)
namespace parser {
namespace x3 = boost::spirit::x3;

template <class T>
struct TrailingBlanksRuleTag;

template <class T, class Tag = TrailingBlanksRuleTag<T>>
struct TrailingBlanksRule: public x3::rule<Tag, T> {
  using Base = x3::rule<Tag, T>;
  using Base::Base;

  constexpr TrailingBlanksRule(): Base("") { }

  template <class RHS>
  // NOLINTNEXTLINE(misc-unconventional-assign-operator,cppcoreguidelines-c-copy-assignment-signature)
  constexpr auto operator=(const RHS &rhs) const && {
    return Base::operator=(rhs >> +x3::omit[x3::blank]);
  }
};

constexpr auto nonblank_trailing_blanks =
    TrailingBlanksRule<std::string, struct nonblank_trailing_blanks_tag>() =
        +~x3::blank;

constexpr auto double_trailing_blanks = TrailingBlanksRule<double>() =
    x3::double_;

constexpr auto uint_trailing_blanks = TrailingBlanksRule<unsigned int>() =
    x3::uint_;
}  // namespace parser
// NOLINTEND(readability-identifier-naming,*-unused-const-variable)

struct NameMapEntry {
  int first_idx;
  int count = 1;
  std::string safe_name = {};  // NOLINT(readability-redundant-member-init)
};

template <class C, class NameFunc,
          class NameTemp = std::invoke_result_t<NameFunc, int>>
// NOLINTNEXTLINE(clang-diagnostic-unused-template)
std::vector<std::string> make_names_unique(const C &cont, NameFunc ith_name) {
  std::vector<std::string> names;
  names.reserve(cont.size());

  absl::flat_hash_map<NameTemp, NameMapEntry> name_map;
  name_map.reserve(cont.size());

  for (int i = 0; i < cont.size(); ++i) {
    NameTemp name = ith_name(i);

    if (name.empty()) {
      names.push_back({});
      continue;
    }

    auto [it, first] = name_map.try_emplace(
        std::move(name), NameMapEntry { static_cast<int>(names.size()) });
    if (first) {
      it->second.safe_name = internal::ascii_safe(it->first);
      names.push_back(it->second.safe_name);
      continue;
    }

    if (it->second.count == 1)
      names[it->second.first_idx] = absl::StrCat(it->second.safe_name, "1");

    names.push_back(absl::StrCat(it->second.safe_name, ++it->second.count));
  }

  return names;
}
}  // namespace
}  // namespace nuri
#endif /* NURI_FMT_FMT_INTERNAL_H_ */
