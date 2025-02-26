//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_FMT_INTERNAL_H_
#define NURI_FMT_FMT_INTERNAL_H_

#include <cstddef>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/log/absl_check.h>
#include <boost/spirit/home/x3.hpp>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"
#include "nuri/utils.h"

namespace nuri {
namespace internal {
template <auto AltlocExtractor, class AT, class CoordConverter>
void pdb_update_confs(Molecule &mol, const std::vector<AT> &atom_data,
                      const CoordConverter &coords) {
  std::vector<std::string_view> s, t;
  std::vector<std::string_view> *altlocs = &s, *buf = &t;

  for (const AT &ad: atom_data) {
    if (ad.data().size() <= 1)
      continue;

    buf->clear();
    std::set_union(altlocs->cbegin(), altlocs->cend(),
                   make_transform_iterator<AltlocExtractor>(ad.data().begin()),
                   make_transform_iterator<AltlocExtractor>(ad.data().end()),
                   std::back_inserter(*buf));
    std::swap(altlocs, buf);
  }

  if (altlocs->empty()) {
    Matrix3Xd &conf = mol.confs().emplace_back(Matrix3Xd(3, atom_data.size()));
    for (int i = 0; i < atom_data.size(); ++i)
      coords(atom_data[i].first(), conf.col(i));
    return;
  }

  ABSL_DCHECK(altlocs->size() > 1);

  mol.confs().resize(altlocs->size());
  for (int i = 0; i < altlocs->size(); ++i)
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

    if (ad.data().size() == altlocs->size()) {
      for (int j = 0; j < mol.confs().size(); ++j)
        coords(ad.data()[j], mol.confs()[j].col(i));
      continue;
    }

    Vector3d major_coord;
    coords(ad.data()[ad.major()], major_coord);

    int j = 0;
    for (int k = 0; k < ad.data().size(); ++j) {
      ABSL_DCHECK(j < mol.confs().size());

      std::string_view altloc = (*altlocs)[j];
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

constexpr int kChainIdx = 0;

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

  std::vector<std::pair<std::string_view, std::vector<int>>> chains;

  for (int i = 0; i < atoms.size(); ++i) {
    const auto &id = atoms[i].first().id().res;

    auto [cit, _] = insert_sorted(chains, { std::invoke(chain_sv, id), {} },
                                  [](const auto &a, const auto &b) {
                                    return a.first < b.first;
                                  });
    cit->second.push_back(i);
  }

  subs.reserve(residues.size() + chains.size());
  subs.resize(residues.size(), mol.substructure(SubstructCategory::kResidue));
  subs.resize(residues.size() + chains.size(),
              mol.substructure(SubstructCategory::kChain));

  auto sit = subs.begin();
  for (int i = 0; i < residues.size(); ++i, ++sit) {
    auto &data = residues[i];

    sit->update(std::move(data.idxs), {});
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

    ABSL_DCHECK(sit->props()[kChainIdx].first == "chain");
  }

  for (int i = 0; i < chains.size(); ++i, ++sit) {
    auto &chain = chains[i];
    sit->update(std::move(chain.second), {});
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
}  // namespace
}  // namespace nuri
#endif /* NURI_FMT_FMT_INTERNAL_H_ */
