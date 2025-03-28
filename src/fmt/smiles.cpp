//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/smiles.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <queue>
#include <stack>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/attributes.h>
#include <absl/base/optimization.h>
#include <absl/container/fixed_array.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/str_cat.h>
#include <absl/types/span.h>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/deque.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/spirit/home/x3.hpp>

#include "nuri/eigen_config.h"
#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/utils.h"

namespace nuri {
bool SmilesReader::getnext(std::vector<std::string> &block) {
  if (block.empty()) {
    block.emplace_back();
  }

  std::string &smiles = block[0];
  while (std::getline(*is_, smiles) && smiles.empty()) { }
  return static_cast<bool>(*is_);
}

const bool SmilesReaderFactory::kRegistered =
    register_reader_factory<SmilesReaderFactory>({ "smi", "smiles" });

namespace {
namespace x3 = boost::spirit::x3;

enum class Chirality {
  kNone,
  kCW,
  kCCW,
};

// NOLINTNEXTLINE(clang-diagnostic-unused-function)
constexpr int char_to_int(char c) {
  return c - '0';
}

char flip_bond_updown(char orig) {
  switch (orig) {
  case '/':
    return '\\';
  case '\\':
    return '/';
  default:
    ABSL_UNREACHABLE();
  }
}

// NOLINTBEGIN(readability-identifier-naming,clang-diagnostic-unused-template)
// NOLINTBEGIN(clang-diagnostic-unneeded-internal-declaration)

namespace parser {
const struct aliphatic_organic_: public x3::symbols<const Element *> {
  aliphatic_organic_() {
    for (const std::string_view symbol:
         { "Cl", "Br", "B", "C", "N", "O", "F", "P", "S", "I", "*" }) {
      const Element *elem = kPt.find_element(symbol);
      // GCOV_EXCL_START
      ABSL_DCHECK(elem != nullptr) << "Element not found: " << symbol;
      // GCOV_EXCL_STOP
      add(symbol, elem);
    }
  }
} aliphatic_organic;

const struct aromatic_organic_: public x3::symbols<const Element *> {
  aromatic_organic_() {
    for (const std::string_view symbol: { "B", "C", "N", "O", "P", "S" }) {
      const Element *elem = kPt.find_element(symbol);
      // GCOV_EXCL_START
      ABSL_DCHECK(elem != nullptr) << "Element not found: " << symbol;
      // GCOV_EXCL_STOP
      add(absl::AsciiStrToLower(symbol), elem);
    }
  }
} aromatic_organic;

const struct element_symbol_: public x3::symbols<const Element *> {
  element_symbol_() {
    // General element symbols
    for (const Element &e: kPt) {
      add(e.symbol(), &e);
    }

    // Dummy atom
    add("*", &kPt[0]);
  }
} element_symbol;

const struct aromatic_symbol_: public x3::symbols<const Element *> {
  aromatic_symbol_() {
    // Aromatic symbols
    for (const std::string_view symbol:
         { "Se", "As", "B", "C", "N", "O", "P", "S" }) {
      const Element *elem = kPt.find_element(symbol);
      // GCOV_EXCL_START
      ABSL_DCHECK(elem != nullptr) << "Element not found: " << symbol;
      // GCOV_EXCL_STOP
      add(absl::AsciiStrToLower(symbol), elem);
    }
  }
} aromatic_symbol;

const struct chirality_: public x3::symbols<Chirality> {
  chirality_() {
    add("@", Chirality::kCCW)   //
        ("@@", Chirality::kCW)  //
        ;
  }
} chirality;

struct RingData {
  char bond_data;
  int atom_idx;
};

struct mutator_tag;
struct has_hydrogens_tag;
struct last_atom_stack_tag;
struct last_bond_data_tag;
struct ring_map_tag;
struct ring_bonds_tag;
struct bond_geometry_tag;
struct implicit_aromatics_tag;

using HydrogenIdx = std::vector<int>;
using ImplicitAromatics = std::vector<int>;
using AtomIdxStack = std::stack<int, std::vector<int>>;
using RingMap = absl::flat_hash_map<int, RingData>;
using RingBonds = std::vector<int>;
using BondGeometryMap =
    absl::flat_hash_map<int, std::vector<std::pair<int, char>>>;

template <class Ctx>
int get_last_idx(Ctx &ctx) {
  return x3::get<last_atom_stack_tag>(ctx).get().top();
}

template <class Ctx>
int set_last_idx(Ctx &ctx, int idx) {
  return x3::get<last_atom_stack_tag>(ctx).get().top() = idx;
}

constexpr auto update_hydrogen = [](auto &ctx) {
  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
  const int last_idx = get_last_idx(ctx),
            h_count = char_to_int(x3::_attr(ctx).value_or('1'));

  ABSL_DVLOG(3) << "Adding " << h_count << " hydrogens to atom " << last_idx;
  mutator.mol().atom(last_idx).data().set_implicit_hydrogens(h_count);
};

constexpr auto hydrogen = (x3::lit('H') >> -x3::digit)[update_hydrogen];

template <class Ctx>
void set_charge(Ctx &ctx, bool positive, int abs_charge) {
  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
  const int last_idx = get_last_idx(ctx),
            charge = positive ? abs_charge : -abs_charge;
  mutator.mol().atom(last_idx).data().set_formal_charge(charge);

  ABSL_DVLOG(3) << "Setting charge of atom " << last_idx << " to " << charge;
}

constexpr auto update_charge_number = [](auto &ctx) {
  using boost::fusion::at_c;

  set_charge(ctx, at_c<0>(x3::_attr(ctx)) == '+', at_c<1>(x3::_attr(ctx)));
};

constexpr auto update_charge_pluses = [](auto &ctx) {
  set_charge(ctx, true, static_cast<int>(x3::_attr(ctx).size()));
};

constexpr auto update_charge_minuses = [](auto &ctx) {
  set_charge(ctx, false, static_cast<int>(x3::_attr(ctx).size()));
};

const auto charge = (x3::char_("+-") >> x3::uint_)[update_charge_number]
                    | (+x3::char_('+'))[update_charge_pluses]
                    | (+x3::char_('-'))[update_charge_minuses];

constants::BondOrder char_to_bond(char b) {
  switch (b) {
  case '/':
  case '\\':
  case '-':
    return constants::kSingleBond;
  case '=':
    return constants::kDoubleBond;
  case '#':
    return constants::kTripleBond;
  case '$':
    return constants::kQuadrupleBond;
  case ':':
    return constants::kAromaticBond;
    // GCOV_EXCL_START
  default:
    ABSL_UNREACHABLE();
    // GCOV_EXCL_STOP
  }
}

int add_bond(MoleculeMutator &mutator, ImplicitAromatics &aromatics,
             const int prev, const int curr, const char bond_repr) {
  if (prev == curr)
    return -1;

  BondData bond_data;

  // Automatic bond or up/down bond
  bool implicit_aromatic;
  if (bond_repr == '\0' || bond_repr == '\\' || bond_repr == '/') {
    const AtomData &last_atom_data = mutator.mol().atom(prev).data(),
                   &atom_data = mutator.mol().atom(curr).data();
    implicit_aromatic = last_atom_data.is_aromatic() && atom_data.is_aromatic();
    bond_data.order() = implicit_aromatic ? constants::kAromaticBond
                                          : constants::kSingleBond;
  } else {
    implicit_aromatic = false;
    bond_data.order() = char_to_bond(bond_repr);
  }

  ABSL_DVLOG(3) << "Trying to add bond " << prev << " -> " << curr << ": "
                << bond_data.order();

  auto [eid, success] = mutator.add_bond(prev, curr, bond_data);
  if (implicit_aromatic)
    aromatics.push_back(eid);
  return success ? eid : -1;
}

template <class Ctx>
int add_atom(Ctx &ctx, const Element *elem, bool aromatic) {
  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);

  const int idx = mutator.add_atom(AtomData(*elem));
  mutator.mol().atom(idx).data().set_aromatic(aromatic);

  ABSL_DVLOG(3) << "Adding " << (aromatic ? "aromatic " : "") << "atom " << idx
                << " (" << elem->symbol() << ')';

  const int last_idx = get_last_idx(ctx);
  const char last_bond_data = x3::get<last_bond_data_tag>(ctx);
  if (ABSL_PREDICT_TRUE(last_bond_data != '.')) {
    ImplicitAromatics &aromatics = x3::get<implicit_aromatics_tag>(ctx);
    const bool success =
        add_bond(mutator, aromatics, last_idx, idx, last_bond_data) >= 0;

    if (ABSL_PREDICT_FALSE(!success)) {
      x3::_pass(ctx) = false;
      ABSL_LOG(WARNING)
          << "Failed to add bond from " << last_idx << " to " << idx;
      return -1;
    }

    if (last_bond_data == '/' || last_bond_data == '\\') {
      BondGeometryMap &bond_geometry_map = x3::get<bond_geometry_tag>(ctx);
      bond_geometry_map[last_idx].push_back({ idx, last_bond_data });
      bond_geometry_map[idx].push_back(
          { last_idx, flip_bond_updown(last_bond_data) });
    }
  }

  set_last_idx(ctx, idx);
  return idx;
}

template <class Ctx, class Payload>
void add_bracket_atom(Ctx &ctx, const Payload &payload, bool aromatic) {
  using boost::fusion::at_c;

  const int idx = add_atom(ctx, at_c<1>(payload), aromatic);
  if (ABSL_PREDICT_FALSE(idx < 0)) {
    // Failed to add atom.
    return;
  }

  x3::get<has_hydrogens_tag>(ctx).get().push_back(idx);

  auto &maybe_isotope = at_c<0>(x3::_attr(ctx));
  if (maybe_isotope) {
    MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
    mutator.mol().atom(idx).data().set_isotope(*maybe_isotope);
  }
}

constexpr auto bracket_atom_adder(bool is_aromatic) {
  return [is_aromatic](auto &ctx) {
    add_bracket_atom(ctx, x3::_attr(ctx), is_aromatic);
  };
}

constexpr auto update_chirality = [](auto &ctx) {
  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
  const int idx = get_last_idx(ctx);
  const Chirality chiral_kind = x3::_attr(ctx);
  ABSL_DVLOG(3) << "Setting chirality of atom " << idx << " to "
                << static_cast<int>(chiral_kind);

  if (chiral_kind == Chirality::kCW) {
    mutator.mol().atom(idx).data().add_flags(AtomFlags::kChiral
                                             | AtomFlags::kClockWise);
  } else if (chiral_kind == Chirality::kCCW) {
    mutator.mol().atom(idx).data().add_flags(AtomFlags::kChiral);
  }
};

constexpr auto set_atom_class = [](auto &) {
  ABSL_LOG(WARNING) << "Atom classes are currently not implemented";
};

const auto bracket_atom =  //
    x3::lit('[')
    >> ((-x3::uint_ >> element_symbol)[bracket_atom_adder(false)]
        | (-x3::uint_ >> aromatic_symbol)[bracket_atom_adder(true)])
    >> -chirality[update_chirality] >> -hydrogen >> -charge
    >> -(x3::lit(':') >> x3::int_)[set_atom_class] >> x3::lit(']');

constexpr auto set_last_bond_data = [](auto &ctx) {
  x3::get<last_bond_data_tag>(ctx).get() = x3::_attr(ctx);
  ABSL_DVLOG(3) << "Setting last bond data to " << x3::_attr(ctx);
};

constexpr auto set_last_bond_auto = [](auto &ctx) {
  x3::get<last_bond_data_tag>(ctx).get() = '\0';
  ABSL_DVLOG(3) << "Setting last bond data to auto";
};

const auto bond = x3::char_("-=#$:/\\")[set_last_bond_data];

constexpr auto dot = x3::char_('.')[set_last_bond_data];

constexpr auto implicit_bond = x3::eps[set_last_bond_auto];

template <class Ctx>
void handle_ring(Ctx &ctx, int ring_idx) {
  const int current_idx = get_last_idx(ctx);
  const char bond_data = x3::get<last_bond_data_tag>(ctx);
  RingMap &map = x3::get<ring_map_tag>(ctx);
  auto [it, is_new] = map.insert({
      ring_idx, { bond_data, current_idx }
  });

  // New ring index, nothing to do.
  if (is_new) {
    ABSL_DVLOG(3) << "Adding ring index " << ring_idx
                  << ", src: " << current_idx << ", " << bond_data;
    return;
  }

  // Ring index already exists, add a bond.
  const RingData &data = it->second;

  char resolved_bond_data;

  if (data.bond_data == '/' || data.bond_data == '\\' || bond_data == '/'
      || bond_data == '\\') {
    if (bond_data == '=' || bond_data == '#' || bond_data == '$'
        || data.bond_data == '=' || data.bond_data == '#'
        || data.bond_data == '$') {
      x3::_pass(ctx) = false;
      ABSL_LOG(WARNING)
          << "Conflicting ring bond specification: " << data.bond_data << " vs "
          << bond_data;
      return;
    }

    if (data.bond_data == '/' || data.bond_data == '\\') {
      BondGeometryMap &bond_geometry_map = x3::get<bond_geometry_tag>(ctx);
      bond_geometry_map[data.atom_idx].push_back(
          { current_idx, data.bond_data });
    }
    if (bond_data == '/' || bond_data == '\\') {
      BondGeometryMap &bond_geometry_map = x3::get<bond_geometry_tag>(ctx);
      bond_geometry_map[current_idx].push_back({ data.atom_idx, bond_data });
    }

    resolved_bond_data = '\0';
  } else if (data.bond_data == bond_data || bond_data == '\0') {
    resolved_bond_data = data.bond_data;
  } else if (data.bond_data == '\0') {
    resolved_bond_data = bond_data;
  } else {
    // Invalid bond specification.
    x3::_pass(ctx) = false;
    ABSL_LOG(WARNING)
        << "Conflicting ring bond specification: " << data.bond_data << " vs "
        << bond_data;
    return;
  }

  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
  ImplicitAromatics &aromatics = x3::get<implicit_aromatics_tag>(ctx);
  const int bond_id = add_bond(mutator, aromatics, data.atom_idx, current_idx,
                               resolved_bond_data);
  if (ABSL_PREDICT_FALSE(bond_id < 0)) {
    x3::_pass(ctx) = false;
    ABSL_LOG(WARNING) << "Failed to add ring bond from " << data.atom_idx
                      << " to " << current_idx;
    return;
  }

  map.erase(it);

  RingBonds &ring_bonds = x3::get<ring_bonds_tag>(ctx);
  ring_bonds.push_back(bond_id);
}

constexpr auto set_ring_digit = [](auto &ctx) {
  handle_ring(ctx, char_to_int(x3::_attr(ctx)));
};

constexpr auto set_ring_digits = [](auto &ctx) {
  using boost::fusion::at_c;
  auto &digits = x3::_attr(ctx);
  handle_ring(ctx,
              char_to_int(at_c<0>(digits)) * 10 + char_to_int(at_c<1>(digits)));
};

const auto ring_bond = (bond | implicit_bond)
                       >> (x3::digit[set_ring_digit]
                           | '%' >> (x3::digit >> x3::digit)[set_ring_digits]);

constexpr auto organic_atom_adder(bool is_aromatic) {
  return
      [is_aromatic](auto &ctx) { add_atom(ctx, x3::_attr(ctx), is_aromatic); };
}

const auto atom = aliphatic_organic[organic_atom_adder(false)]
                  | aromatic_organic[organic_atom_adder(true)] | bracket_atom;

constexpr x3::rule<class smiles_tag> smiles = "smiles";

constexpr auto push_last_idx = [](auto &ctx) {
  x3::get<last_atom_stack_tag>(ctx).get().push(get_last_idx(ctx));
};

constexpr auto pop_last_idx = [](auto &ctx) {
  x3::get<last_atom_stack_tag>(ctx).get().pop();
};

const auto branch = x3::lit('(')[push_last_idx] >> (bond | dot | implicit_bond)
                    >> smiles >> x3::lit(')')[pop_last_idx];

const auto branched_atom = atom >> *ring_bond >> *branch;

const auto bond_branched_atom = -(bond | dot) >> branched_atom;

const auto smiles_def = branched_atom >> *bond_branched_atom;

BOOST_SPIRIT_DEFINE(smiles)
}  // namespace parser

// NOLINTEND(clang-diagnostic-unneeded-internal-declaration)
// NOLINTEND(readability-identifier-naming,clang-diagnostic-unused-template)

char resolve_bond_updown_atom(Molecule::Atom atom, Molecule::Atom other,
                              const std::vector<std::pair<int, char>> &cfgs) {
  // both bonds are annotated, so first one is the first neighbor (degree <= 3)
  if (cfgs.size() == 2)
    return cfgs[0].second;

  auto first_nei = absl::c_find_if(atom, [&](Molecule::Neighbor nei) {
    return nei.dst().id() != other.id();
  });
  ABSL_DCHECK(!first_nei.end());

  if (cfgs[0].first == first_nei->dst().id())
    return cfgs[0].second;

  return flip_bond_updown(cfgs[0].second);
}

bool update_bond_configuration(Molecule &mol,
                               const parser::BondGeometryMap &configs) {
  if (configs.empty())
    return true;

  // NOLINTNEXTLINE(readability-use-anyofallof)
  for (auto bond: mol.bonds()) {
    if (bond.data().order() != constants::kDoubleBond
        || all_neighbors(bond.src()) > 3 || all_neighbors(bond.dst()) > 3) {
      continue;
    }

    auto it = configs.find(bond.src().id()), jt = configs.find(bond.dst().id());
    if (it == configs.end() || jt == configs.end())
      continue;

    auto &src_cfgs = it->second, &dst_cfgs = jt->second;
    const bool src_empty = src_cfgs.empty(), dst_empty = dst_cfgs.empty();
    ABSL_ASSUME(!src_empty && !dst_empty);
    if (src_cfgs.size() > 2 || dst_cfgs.size() > 2) {
      ABSL_LOG(INFO) << "too many bond configurations specified on bond "
                     << bond.id() << "; ignoring";
      continue;
    }

    if ((src_cfgs.size() == 2 && src_cfgs[0].second == src_cfgs[1].second)
        || (dst_cfgs.size() == 2 && dst_cfgs[0].second == dst_cfgs[1].second)) {
      ABSL_LOG(WARNING)
          << "conflicting bond configuration specified on bond " << bond.id();
      return false;
    }

    const char src_cfg =
                   resolve_bond_updown_atom(bond.src(), bond.dst(), src_cfgs),
               dst_cfg =
                   resolve_bond_updown_atom(bond.dst(), bond.src(), dst_cfgs);

    // C/C=C/C or C\C=C\C  // NOLINT
    bool trans = src_cfg != dst_cfg;
    bond.data().add_flags(BondFlags::kConfigSpecified).set_trans(trans);
  }

  return true;
}

void update_implicit_hydrogens(Molecule::MutableAtom atom) {
  int sum_bo = internal::sum_bond_order_raw(atom, 0, true), normal_valence;

  switch (atom.data().atomic_number()) {
  case 0:
    // Dummy atom: skip
    return;
  case 5:
  case 7:
    // Boron, nitrogen: 3 bonds
    normal_valence = 3;
    break;
  case 6:
    // Carbon: 4 bonds
    normal_valence = 4;
    break;
  case 8:
    // Oxygen: 2 bonds
    normal_valence = 2;
    break;
  case 15:
    // Phosphorus: 3 or 5 bonds
    normal_valence = sum_bo > 3 ? 5 : 3;
    break;
  case 16:
    // Sulfur: 2, 4 or 6 bonds
    normal_valence = sum_bo > 2 ? (sum_bo > 4 ? 6 : 4) : 2;
    break;
  default:
    // Halogens, 1 bond
    normal_valence = 1;
  }

  atom.data().set_implicit_hydrogens(nonnegative(normal_valence - sum_bo));
}

/**
 * Smiles orders bonds in this order:
 *   1) Previous atom (max 1)
 *   2) Implicit hydrogen (max 1 if chiral)
 *   3) Broken bonds
 *   4) Others
 *
 * On the other hand, we order bonds in this order:
 *   1) we place implicit hydrogen (if present) last. This is to preserve
 *      chirality when the implicit hydrogen is made explicit.
 *   2) we place broken bonds at unknown order. This is because the broken bonds
 *      could not be added before the partner atom is added.
 */
template <class Pred>
void smiles_ordering_chirality(Molecule::Atom atom, std::vector<int> &ordering,
                               const Pred &is_broken_bond) {
  ordering.clear();

  // 1) Previous atom (must be the first neighbor if present)
  if (atom[0].dst().id() < atom.id() && !is_broken_bond(atom[0].eid()))
    ordering.push_back(0);

  // 2) Implicit hydrogen
  if (atom.data().implicit_hydrogens() > 0)
    ordering.push_back(atom.degree());

  // 3) Broken bonds
  for (int i = 0; i < atom.degree(); ++i) {
    auto nei = atom[i];
    if (is_broken_bond(nei.eid()))
      ordering.push_back(i);
  }

  // 4) Others (always id > atom.id())
  for (int i = 0; i < atom.degree(); ++i) {
    auto nei = atom[i];
    if (nei.dst().id() > atom.id() && !is_broken_bond(nei.eid()))
      ordering.push_back(i);
  }
}

/**
 * If i and ordering[i] is consistent in terms of chirality resolution, return
 * true.
 */
bool chiral_order_consistent(const std::vector<int> &ordering) {
  const auto ord_size = ordering.size();
  ABSL_ASSUME(ord_size == 4);

  // Handedness change by axis. 0, 2 -> same, 1, 3 -> opposite
  bool consistent = ordering[0] % 2 == 0;

  auto nbs = absl::MakeConstSpan(ordering).subspan(1);
  const int start = static_cast<int>(absl::c_min_element(nbs) - nbs.begin());
  const int n = static_cast<int>(nbs.size());

  // Now check if order is consistent
  for (int i = 1; i < n; ++i) {
    int left = (start + i - 1) % n;
    int right = (start + i) % n;
    if (nbs[left] > nbs[right]) {
      consistent = !consistent;
      break;
    }
  }

  return consistent;
}

/**
 * Resolve chirality of the atom based on the "SMILES ordering" of neighbors.
 * Will pass-through the current chirality if the atom has less than 3 neighbors
 * or has more than 1 implicit hydrogens (not a stereocenter).
 */
template <class Pred>
bool smiles_resolve_is_clockwise(Molecule::Atom atom,
                                 std::vector<int> &ordering,
                                 Pred &&is_ring_bond) {
  if (atom.degree() < 3) {
    ABSL_LOG(INFO) << "Atom " << atom.id() << " has degree " << atom.degree()
                   << " (< 3), but is marked chiral";
    return atom.data().is_clockwise();
  }

  if (atom.data().implicit_hydrogens() > 1) {
    ABSL_LOG(INFO)
        << "Atom " << atom.id() << " has " << atom.data().implicit_hydrogens()
        << " implicit hydrogens (> 1), but is marked chiral";
    return atom.data().is_clockwise();
  }

  smiles_ordering_chirality(atom, ordering, std::forward<Pred>(is_ring_bond));

  if (ordering.size() != 4) {
    ABSL_LOG(INFO) << "Atom " << atom.id() << " has total " << ordering.size()
                   << " neighbors (!= 4), but is marked chiral";
    return atom.data().is_clockwise();
  }

  const bool consistent = chiral_order_consistent(ordering);
  // consistent (T) and CW (T) -> CW (T)
  // consistent (T) and CCW (F) -> CCW (F)
  // inconsistent (F) and CW (T) -> CCW (F)
  // inconsistent (F) and CCW (F) -> CW (T)
  return consistent == atom.data().is_clockwise();
}

void convert_chirality(Molecule &mol, const parser::RingBonds &ring_bonds) {
  ArrayXb is_ring_bond = ArrayXb::Zero(mol.num_bonds());
  is_ring_bond(ring_bonds) = true;

  std::vector<int> smiles_order;
  smiles_order.reserve(4);

  for (auto atom: mol) {
    if (!atom.data().is_chiral())
      continue;

    bool clockwise =
        smiles_resolve_is_clockwise(atom, smiles_order, is_ring_bond);
    atom.data().set_clockwise(clockwise);
  }
}
}  // namespace

Molecule read_smiles(const std::vector<std::string> &smi_block) {
  Molecule mol;
  if (smi_block.empty()) {
    ABSL_LOG(WARNING) << "Empty SMILES block";
    return mol;
  }

  const std::string &smiles = smi_block[0];

  // Context variables
  parser::HydrogenIdx has_hydrogens;
  parser::AtomIdxStack stack;
  char last_bond_data = '.';
  parser::RingMap ring_map;
  parser::RingBonds ring_bonds;
  parser::BondGeometryMap bond_geometry_map;
  parser::ImplicitAromatics implicit_aromatics;

  stack.push(-1);

  auto begin = smiles.begin();
  {
    MoleculeMutator mutator = mol.mutator();

    // clang-format off
    auto parser = x3::with<parser::mutator_tag>(std::ref(mutator))
        [x3::with<parser::has_hydrogens_tag>(std::ref(has_hydrogens))
        [x3::with<parser::last_atom_stack_tag>(std::ref(stack))
        [x3::with<parser::last_bond_data_tag>(std::ref(last_bond_data))
        [x3::with<parser::ring_map_tag>(std::ref(ring_map))
        [x3::with<parser::ring_bonds_tag>(std::ref(ring_bonds))
        [x3::with<parser::bond_geometry_tag>(std::ref(bond_geometry_map))
        [x3::with<parser::implicit_aromatics_tag>(std::ref(implicit_aromatics))
        [parser::smiles]]]]]]]];
    // clang-format on

    bool success = x3::parse_main(begin, smiles.end(), parser, x3::unused);

    if (success && !ring_map.empty()) {
      ABSL_LOG(WARNING) << "Unresolved ring bonds: " << ring_map.size();
      success = false;
    }

    if (!success) {
      ABSL_LOG(ERROR) << "Parsing failed: " << smiles;
      mutator.clear();
      return mol;
    }
  }

  for (int bid: implicit_aromatics) {
    if (!mol.bond(bid).data().is_ring_bond()) {
      mol.bond(bid)
          .data()
          .set_order(constants::kSingleBond)
          .del_flags(BondFlags::kAromatic);
    }
  }

  if (!update_bond_configuration(mol, bond_geometry_map)) {
    ABSL_LOG(ERROR) << "Parsing failed: " << smiles;
    mol.clear();
    return mol;
  }

  auto hit = has_hydrogens.begin();
  for (auto atom: mol) {
    if (hit != has_hydrogens.end() && *hit == atom.id()) {
      ++hit;
      continue;
    }

    update_implicit_hydrogens(atom);
  }

  convert_chirality(mol, ring_bonds);

  while (begin != smiles.end() && std::isspace(*begin) != 0)
    ++begin;
  mol.name() = std::string_view(&*begin, smiles.end() - begin);

  return mol;
}

namespace {
void break_cycles(const Molecule &mol, ArrayXi &atom_order,
                  ArrayXi &bond_visited, std::vector<int> &broken_bonds,
                  absl::InlinedVector<int, 1> &roots) {
  int atom_idx = 0;
  auto dfs = [&](auto &self, Molecule::Atom atom, int prev) -> void {
    atom_order[atom.id()] = atom_idx++;

    for (auto nei: atom) {
      if (nei.dst().id() == prev)
        continue;

      if (atom_order[nei.dst().id()] >= 0) {
        if (bond_visited[nei.eid()] == 0) {
          bond_visited[nei.eid()] = 1;
          broken_bonds.push_back(nei.eid());
        }
        continue;
      }

      bond_visited[nei.eid()] = 1;
      self(self, nei.dst(), atom.id());
    }
  };

  for (Molecule::Atom atom: mol) {
    if (atom_order[atom.id()] >= 0)
      continue;

    roots.push_back(atom.id());
    dfs(dfs, atom, -1);

    if (atom_idx == mol.num_atoms())
      break;
  }
}

bool number_rings(ArrayXi &ring_idxs, const Molecule &mol,
                  const ArrayXi &atom_order,
                  const std::vector<int> &broken_bonds) {
  ArrayXi left(broken_bonds.size()), right(broken_bonds.size());
  for (int i = 0; i < broken_bonds.size(); ++i) {
    auto bond = mol.bond(broken_bonds[i]);
    std::tie(left[i], right[i]) =
        nuri::minmax(atom_order[bond.src().id()], atom_order[bond.dst().id()]);
  }

  ArrayXi left_idxs_ordered = argsort<>(left);
  if (broken_bonds.size() < 100) {
    for (int i = 0; i < left.size(); ++i)
      ring_idxs[broken_bonds[left_idxs_ordered[i]]] = i + 1;

    return true;
  }

  ArrayXi right_idxs_ordered = argsort<>(right);
  std::priority_queue ring_idxs_avail(boost::make_counting_iterator(1),
                                      boost::make_counting_iterator(100),
                                      std::greater<>());
  int j = 0;
  for (int lp: left_idxs_ordered) {
    for (; right[right_idxs_ordered[j]] < left[lp]; ++j) {
      int rp = right_idxs_ordered[j];
      ring_idxs_avail.push(ring_idxs[broken_bonds[rp]]);
    }

    if (ring_idxs_avail.empty())
      return false;

    int idx = ring_idxs_avail.top();
    ring_idxs_avail.pop();
    ring_idxs[broken_bonds[lp]] = idx;
  }

  return true;
}

enum class BondConfig : std::uint8_t {
  kNone = 0,
  // xor 1 to flip up/down
  kUp = 0b10,
  kDown = 0b11,
};

BondConfig flip_updown(BondConfig kind) {
  ABSL_DCHECK(kind == BondConfig::kUp || kind == BondConfig::kDown);
  return kind ^ static_cast<BondConfig>(0b1);
}

BondConfig updown_other(BondConfig kind, bool trans) {
  ABSL_DCHECK(kind == BondConfig::kUp || kind == BondConfig::kDown);
  return trans ? flip_updown(kind) : kind;
}

class NeighborBondConfig {
public:
  // NOLINTNEXTLINE(clang-diagnostic-unused-member-function)
  NeighborBondConfig(Molecule::Atom atom)
      : cfgs_(atom.degree(), BondConfig::kNone) {
    for (int i = 0; i < atom.degree(); ++i) {
      auto nei = atom[i];
      if (nei.edge_data().has_config()) {
        ++nconf_;
        first_config_ = i;
      } else if ((atom[i].edge_data().order() == constants::kSingleBond
                  || atom[i].edge_data().order() == constants::kAromaticBond)) {
        if (free_nei_[0] < 0) {
          free_nei_[0] = i;
        } else if (free_nei_[1] < 0) {
          free_nei_[1] = i;
        }
      }
    }
  }

  BondConfig src_updown(Molecule::Atom src, BondConfig kind) {
    ABSL_DCHECK_EQ(nconf_, 1);
    ABSL_DCHECK_GE(free_nei_[0], 0);

    if (kind == BondConfig::kNone) {
      kind = src.id() < src[first_free()].dst().id() ? BondConfig::kUp
                                                     : BondConfig::kDown;
    }

    cfgs_[first_free()] = kind;
    if (second_free() >= 0)
      cfgs_[second_free()] = flip_updown(kind);

    return kind;
  }

  bool dst_updown(Molecule::Neighbor nei, BondConfig kind) {
    ABSL_DCHECK(kind == BondConfig::kUp || kind == BondConfig::kDown);
    ABSL_DCHECK_GE(free_nei_[0], 0);

    auto src = nei.src(), dst = nei.dst();
    const int srci = dst.find_adjacent(src) - dst.begin();
    ABSL_DCHECK_LT(srci, cfgs_.size());

    if (cfgs_[srci] != BondConfig::kNone && cfgs_[srci] != kind)
      return false;

    cfgs_[srci] = kind;

    if (nconf_ == 1) {
      int other_idx = free_nei_[value_if(srci == first_free())];
      if (other_idx >= 0) {
        kind = flip_updown(kind);
        ABSL_DCHECK(cfgs_[other_idx] == BondConfig::kNone
                    || cfgs_[other_idx] == kind);
        cfgs_[other_idx] = kind;
      }
    }

    return true;
  }

  BondConfig updown() const {
    ABSL_DCHECK_EQ(nconf_, 1);
    ABSL_DCHECK_GE(free_nei_[0], 0);
    return cfgs_[first_free()];
  }

  const std::vector<BondConfig> &cfgs() const { return cfgs_; }

  bool has_free() const { return free_nei_[0] >= 0; }

  int first_free() const { return free_nei_[0]; }

  int second_free() const { return free_nei_[1]; }

  int first_config() const { return first_config_; }

  int nconf() const { return nconf_; }

private:
  std::vector<BondConfig> cfgs_;
  int free_nei_[2] = { -1, -1 };
  int first_config_ = -1;
  int nconf_ = 0;
};

class BondConfigResolver {
public:
  BondConfigResolver(const Molecule &mol, const ArrayXi &ring_idxs)
      : data_(mol.begin(), mol.end()), ring_idxs_(&ring_idxs) { }

  const NeighborBondConfig &cfg(Molecule::Atom atom) const {
    return data_[atom.id()];
  }

  bool resolve_updown(Molecule::Atom atom, BondConfig req) {
    NeighborBondConfig &cfg = mcfg(atom);
    ABSL_DCHECK_EQ(cfg.nconf(), 1);

    if (!cfg.has_free()) {
      ABSL_LOG(WARNING)
          << "no free bond to resolve up/down configuration on atom "
          << atom.id();
      return false;
    }

    if (cfg.updown() != BondConfig::kNone) {
      bool consistent = req == BondConfig::kNone || req == cfg.updown();
      ABSL_LOG_IF(WARNING, !consistent)
          << "conflicting bond up/down configuration specified on atom "
          << atom.id();
      return consistent;
    }

    BondConfig kind = cfg.src_updown(atom, req);

    bool success = dst_updown(atom[cfg.first_free()], flip_updown(kind));

    if (ABSL_PREDICT_TRUE(success) && cfg.second_free() >= 0)
      success = dst_updown(atom[cfg.second_free()], kind);

    return success;
  }

  BondConfig updown(Molecule::Atom atom) const { return cfg(atom).updown(); }

private:
  bool dst_updown(Molecule::Neighbor nei, BondConfig kind) {
    if (is_broken(nei.eid()))
      return true;

    if (!mcfg(nei.dst()).dst_updown(nei, kind)) {
      ABSL_LOG(WARNING)
          << "conflicting bond up/down configuration specified on atom "
          << nei.dst().id();
      return false;
    }

    return true;
  }

  NeighborBondConfig &mcfg(Molecule::Atom atom) { return data_[atom.id()]; }

  bool is_broken(int eid) const { return (*ring_idxs_)[eid] > 0; }

  absl::FixedArray<NeighborBondConfig> data_;
  const ArrayXi *ring_idxs_;
};

bool smiles_resolve_bond_updown(const Molecule &mol,
                                const absl::InlinedVector<int, 1> &roots,
                                ArrayXi &visited,
                                BondConfigResolver &resolver) {
  auto dfs = [&](auto &self, int curr) -> bool {
    visited[curr] = 1;

    auto atom = mol.atom(curr);
    auto &info = resolver.cfg(atom);
    if (info.nconf() > 2) {
      ABSL_LOG(WARNING)
          << "too many bond configurations specified on atom " << curr;
      return false;
    }

    if (info.nconf() == 1) {
      if (all_neighbors(atom) > 3) {
        ABSL_LOG(WARNING) << "cannot determine bond configuration for atom "
                          << curr << " with more than 3 neighbors";
        return false;
      }

      auto cfg_nei = atom[info.first_config()];

      BondConfig req;
      if (visited[cfg_nei.dst().id()] == 0
          || !resolver.cfg(cfg_nei.dst()).has_free()) {
        req = BondConfig::kNone;
      } else {
        req = updown_other(resolver.updown(cfg_nei.dst()),
                           cfg_nei.edge_data().is_trans());
      }
      if (!resolver.resolve_updown(atom, req))
        return false;
    }

    for (auto nit = atom.begin(); nit != atom.end(); ++nit) {
      if (visited[nit->dst().id()] != 0)
        continue;

      if (!self(self, nit->dst().id()))
        return false;
    }
    return true;
  };

  for (int root: roots)
    if (!dfs(dfs, root))
      return false;

  ABSL_DCHECK(visited.cast<bool>().all());

  return true;
}

std::string_view smiles_symbol(const AtomData &data) {
  if (ABSL_PREDICT_FALSE(data.atomic_number() == 0))
    return "*";

  return data.element_symbol();
}

std::string_view smiles_symbol(Molecule::Atom atom) {
  return smiles_symbol(atom.data());
}

bool can_write_organic(Molecule::Atom atom) {
  if (atom.data().formal_charge() != 0 || atom.data().is_chiral()
      || ABSL_PREDICT_FALSE(atom.data().explicit_isotope() != nullptr))
    return false;

  // See update_implicit_hydrogens()
  int valence = internal::sum_bond_order_raw(atom, 0, true), pred_implicit_h;

  switch (atom.data().atomic_number()) {
  case 0:
    return true;
  case 5:
  case 7:
    pred_implicit_h = 3 - valence;
    break;
  case 6:
    pred_implicit_h = 4 - valence;
    break;
  case 8:
    pred_implicit_h = 2 - valence;
    break;
  case 15:
    pred_implicit_h = valence > 3 ? valence - 5 : valence - 3;
    break;
  case 16:
    pred_implicit_h = valence > 4   ? 6 - valence
                      : valence > 2 ? 4 - valence
                                    : 2 - valence;
    break;
  case 9:  // halogens
  case 17:
  case 35:
  case 53:
    return valence == 1 && !atom.data().is_aromatic();
  default:
    return false;
  }

  return pred_implicit_h == atom.data().implicit_hydrogens();
}

void write_organic(std::string &out, Molecule::Atom atom) {
  if (atom.data().is_aromatic()) {
    ABSL_DCHECK(smiles_symbol(atom).size() == 1);
    out.push_back(absl::ascii_tolower(smiles_symbol(atom)[0]));
    return;
  }

  absl::StrAppend(&out, smiles_symbol(atom));
}

bool can_write_aromatic_symbol(const AtomData &data) {
  switch (data.atomic_number()) {
  case 5:
  case 6:
  case 7:
  case 8:
  case 15:
  case 16:
  case 33:
  case 34:
    return data.is_aromatic();
  default:
    return false;
  }
}

void write_bracket_atom(std::string &out, Molecule::Atom atom,
                        std::vector<int> &smiles_order,
                        const ArrayXi &ring_idxs) {
  out.push_back('[');

  if (auto iso = atom.data().explicit_isotope(); iso != nullptr)
    absl::StrAppend(&out, iso->mass_number);

  if (can_write_aromatic_symbol(atom.data())) {
    absl::StrAppend(&out, absl::AsciiStrToLower(smiles_symbol(atom)));
  } else {
    absl::StrAppend(&out, smiles_symbol(atom));
  }

  if (atom.data().is_chiral()) {
    const bool clockwise = smiles_resolve_is_clockwise(
        atom, smiles_order, [&](int eid) { return ring_idxs[eid] > 0; });

    out.push_back('@');
    if (clockwise)
      out.push_back('@');
  }

  if (atom.data().atomic_number() != 1) {
    int hcnt = atom.data().implicit_hydrogens();
    if (hcnt > 0)
      out.push_back('H');
    if (hcnt > 1)
      absl::StrAppend(&out, hcnt);
  }

  if (int fchg = atom.data().formal_charge(); fchg != 0) {
    int magnitude = std::abs(fchg);
    out.push_back(fchg > 0 ? '+' : '-');
    if (magnitude > 1)
      absl::StrAppend(&out, magnitude);
  }

  out.push_back(']');
}

void write_atom(std::string &out, Molecule::Atom atom,
                std::vector<int> &smiles_order, const ArrayXi &ring_idxs) {
  if (can_write_organic(atom)) {
    write_organic(out, atom);
  } else {
    write_bracket_atom(out, atom, smiles_order, ring_idxs);
  }
}

char bond_to_char(constants::BondOrder order) {
  switch (order) {
  case constants::kOtherBond:
    ABSL_LOG(WARNING) << "Other bond treated as single bond";
    ABSL_FALLTHROUGH_INTENDED;
  case constants::kSingleBond:
    return '-';
  case constants::kDoubleBond:
    return '=';
  case constants::kTripleBond:
    return '#';
  case constants::kQuadrupleBond:
    return '$';
  case constants::kAromaticBond:
    return ':';
  }

  // GCOV_EXCL_START
  ABSL_UNREACHABLE();
  // GCOV_EXCL_STOP
}

void write_bond_order(std::string &out, Molecule::Neighbor nei) {
  if (can_write_aromatic_symbol(nei.src().data())
      && can_write_aromatic_symbol(nei.dst().data())) {
    if (nei.edge_data().order() == constants::kAromaticBond)
      return;
  } else if (nei.edge_data().order() == constants::kSingleBond) {
    return;
  }

  out.push_back(bond_to_char(nei.edge_data().order()));
}

void smiles_write_bond(std::string &out, Molecule::Atom src, int nidx,
                       const BondConfigResolver &resolver) {
  BondConfig kind = resolver.cfg(src).cfgs()[nidx];
  if (kind == BondConfig::kNone) {
    write_bond_order(out, src[nidx]);
    return;
  }

  ABSL_ASSUME(kind == BondConfig::kUp || kind == BondConfig::kDown);
  char updown = kind == BondConfig::kUp ? '/' : '\\';
  out.push_back(updown);
}

void write_ring_index(std::string &out, int ring_idx) {
  if (ring_idx < 10) {
    out.push_back(static_cast<char>('0' + ring_idx));
  } else {
    absl::StrAppend(&out, "%", ring_idx);
  }
}

void do_write_smiles_simple(std::string &out, const Molecule &mol,
                            const absl::InlinedVector<int, 1> &roots,
                            const ArrayXi &ring_idxs, ArrayXi &atom_visited,
                            const BondConfigResolver &resolver) {
  std::vector<int> smiles_order;
  smiles_order.reserve(4);

  auto write = [&](auto &self, Molecule::Atom atom,
                   Molecule::const_neighbor_iterator prev_it) -> void {
    atom_visited[atom.id()] = 1;

    int prev;
    if (!prev_it.end()) {
      prev = prev_it->src().id();
      smiles_write_bond(out, prev_it->src(), prev_it - prev_it->src().begin(),
                        resolver);
    } else {
      prev = -1;
    }

    write_atom(out, atom, smiles_order, ring_idxs);

    for (auto nit = atom.begin(); nit != atom.end(); ++nit) {
      int ring_idx = ring_idxs[nit->eid()];
      if (ring_idx == 0)
        continue;

      smiles_write_bond(out, atom, nit - atom.begin(), resolver);
      write_ring_index(out, ring_idx);
    }

    auto will_write = [&](Molecule::Neighbor nei) {
      return nei.dst().id() != prev && atom_visited[nei.dst().id()] == 0
             && ring_idxs[nei.eid()] == 0;
    };
    int implicit_hydrogen_write_count =
        atom.data().atomic_number() == 1 ? atom.data().implicit_hydrogens() : 0;

    int branch_count =
        absl::c_count_if(atom, will_write) + implicit_hydrogen_write_count;

    for (int i = 0; i < implicit_hydrogen_write_count; ++i) {
      bool need_parens = --branch_count > 0;
      absl::StrAppend(&out, need_parens ? "([H])" : "[H]");
    }

    for (auto nit = atom.begin(); nit != atom.end(); ++nit) {
      if (!will_write(*nit))
        continue;

      if (--branch_count > 0) {
        out.push_back('(');
        self(self, nit->dst(), nit);
        out.push_back(')');
      } else {
        self(self, nit->dst(), nit);
      }
    }
  };

  bool first = true;
  for (int root: roots) {
    if (!first)
      out.push_back('.');

    write(write, mol.atom(root), mol.atom(root).end());
    first = false;
  }
}

bool write_smiles_simple(std::string &out, const Molecule &mol) {
  ArrayXi atom_order = ArrayXi::Constant(mol.num_atoms(), -1),
          ring_idxs = ArrayXi::Zero(mol.num_bonds());

  std::vector<int> broken_bonds;
  absl::InlinedVector<int, 1> roots;
  break_cycles(mol, atom_order, ring_idxs, broken_bonds, roots);

  ring_idxs.setZero();
  bool can_write_ring = number_rings(ring_idxs, mol, atom_order, broken_bonds);
  if (!can_write_ring) {
    ABSL_LOG(WARNING) << "Ring number exceeds 100";
    return false;
  }

  atom_order.setZero();
  BondConfigResolver resolver(mol, ring_idxs);
  bool can_write_cfg =
      smiles_resolve_bond_updown(mol, roots, atom_order, resolver);
  if (!can_write_cfg)
    return false;

  atom_order.setZero();
  do_write_smiles_simple(out, mol, roots, ring_idxs, atom_order, resolver);
  return true;
}
}  // namespace

bool write_smiles(std::string &out, const Molecule &mol, bool canonical) {
  if (canonical) {
    ABSL_LOG(ERROR) << "Canonical SMILES is not implemented yet";
    return false;
  }

  bool ok = write_smiles_simple(out, mol);
  if (!ok) {
    ABSL_LOG(ERROR) << "Failed to write SMILES";
    return false;
  }

  absl::StrAppend(&out, "\t", internal::ascii_newline_safe(mol.name()), "\n");
  return true;
}
}  // namespace nuri
