//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/smiles.h"

#include <initializer_list>
#include <memory>
#include <stack>
#include <string>
#include <string_view>
#include <vector>

#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/deque.hpp>
#include <boost/spirit/home/x3.hpp>

#include <absl/base/optimization.h>
#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/str_cat.h>

#include "nuri/core/element.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"

namespace nuri {
bool SmilesStream::advance() {
  if (!is_->good()) {
    return false;
  }

  do {
    std::getline(*is_, line_);
  } while (line_.empty() && is_->good());

  return !line_.empty();
}

const bool SmileStreamFactory::kRegistered =
  MoleculeStreamFactory::register_factory(
    std::make_unique<SmileStreamFactory>(), { "smi", "smiles" });

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

// NOLINTBEGIN(readability-identifier-naming,clang-diagnostic-unused-template)
// NOLINTBEGIN(clang-diagnostic-unneeded-internal-declaration)

namespace parser {
const struct aliphatic_organic_: public x3::symbols<const Element *> {
  aliphatic_organic_() {
    for (const std::string_view symbol:
         { "Cl", "Br", "B", "C", "N", "O", "F", "P", "S", "I", "*" }) {
      const Element *elem = PeriodicTable::get().find_element(symbol);
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
      const Element *elem = PeriodicTable::get().find_element(symbol);
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
    for (const Element &e: PeriodicTable::get()) {
      add(e.symbol(), &e);
    }

    // Dummy atom
    add("*", &PeriodicTable::get()[0]);
  }
} element_symbol;

const struct aromatic_symbol_: public x3::symbols<const Element *> {
  aromatic_symbol_() {
    // Aromatic symbols
    for (const std::string_view symbol:
         { "Se", "As", "B", "C", "N", "O", "P", "S" }) {
      const Element *elem = PeriodicTable::get().find_element(symbol);
      // GCOV_EXCL_START
      ABSL_DCHECK(elem != nullptr) << "Element not found: " << symbol;
      // GCOV_EXCL_STOP
      add(absl::AsciiStrToLower(symbol), elem);
    }
  }
} aromatic_symbol;

const struct chirality_: public x3::symbols<Chirality> {
  chirality_() {
    add("@", Chirality::kCCW)  //
      ("@@", Chirality::kCW)   //
      ;
  }
} chirality;

struct RingData {
  char bond_data;
  int atom_idx;
};

struct mutator_tag;
struct last_atom_stack_tag;
struct last_bond_data_tag;
struct chirality_map_tag;
struct ring_map_tag;
struct bond_geometry_tag;

using AtomIdxStack = std::stack<int, std::vector<int>>;
using ChiralityMap = absl::flat_hash_map<int, Chirality>;
using RingMap = absl::flat_hash_map<int, RingData>;
using BondGeometryMap = absl::flat_hash_map<std::pair<int, int>, char>;

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

  ABSL_DLOG(INFO) << "Adding " << h_count << " hydrogens to atom " << last_idx;
  mutator.atom_data(last_idx).set_implicit_hydrogens(h_count);
};

constexpr auto hydrogen = (x3::lit('H') >> -x3::digit)[update_hydrogen];

template <class Ctx>
void set_charge(Ctx &ctx, bool positive, int abs_charge) {
  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
  const int last_idx = get_last_idx(ctx),
            charge = positive ? abs_charge : -abs_charge;
  mutator.atom_data(last_idx).set_formal_charge(charge);

  ABSL_DLOG(INFO) << "Setting charge of atom " << last_idx << " to " << charge;
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
  }

  // GCOV_EXCL_START
  ABSL_UNREACHABLE();
  // GCOV_EXCL_STOP
}

bool add_bond(MoleculeMutator &mutator, const int prev, const int curr,
              const char bond_repr) {
  BondData bond_data;

  // Automatic bond
  if (bond_repr == '\0') {
    const AtomData &last_atom_data = mutator.atom_data(prev),
                   &atom_data = mutator.atom_data(curr);
    bond_data.order() = last_atom_data.is_aromatic() && atom_data.is_aromatic()
                          ? constants::kAromaticBond
                          : constants::kSingleBond;
  } else {
    bond_data.order() = char_to_bond(bond_repr);
  }

  ABSL_DLOG(INFO) << "Trying to add bond " << prev << " -> " << curr << ": "
                  << bond_data.order();

  return mutator.add_bond(prev, curr, bond_data);
}

template <class Ctx>
int add_atom(Ctx &ctx, const Element *elem, int implicit_hydrogens,
             bool aromatic) {
  MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);

  const int idx = mutator.add_atom(AtomData(*elem, implicit_hydrogens));
  mutator.atom_data(idx).set_aromatic(aromatic);

  ABSL_DLOG(INFO) << "Adding " << (aromatic ? "aromatic " : "") << "atom "
                  << idx << " (" << elem->symbol() << ')';

  const int last_idx = get_last_idx(ctx);
  const char last_bond_data = x3::get<last_bond_data_tag>(ctx);
  if (ABSL_PREDICT_TRUE(last_bond_data != '.')) {
    const bool success = add_bond(mutator, last_idx, idx, last_bond_data);

    if (ABSL_PREDICT_FALSE(!success)) {
      x3::_pass(ctx) = false;
      ABSL_LOG(WARNING)
        << "Failed to add bond from " << last_idx << " to " << idx;
      return -1;
    }

    if (last_bond_data == '/' || last_bond_data == '\\') {
      BondGeometryMap &bond_geometry_map = x3::get<bond_geometry_tag>(ctx);
      bond_geometry_map[std::make_pair(last_idx, idx)] = last_bond_data;
    }
  }

  set_last_idx(ctx, idx);
  return idx;
}

template <class Ctx, class Payload>
void add_bracket_atom(Ctx &ctx, const Payload &payload, bool aromatic) {
  using boost::fusion::at_c;

  const int idx = add_atom(ctx, at_c<1>(payload), 0, aromatic);
  if (ABSL_PREDICT_FALSE(idx < 0)) {
    // Failed to add atom.
    return;
  }

  auto &maybe_isotope = at_c<0>(x3::_attr(ctx));
  if (maybe_isotope) {
    MoleculeMutator &mutator = x3::get<mutator_tag>(ctx);
    mutator.atom_data(idx).set_isotope(*maybe_isotope);
  }
}

constexpr auto bracket_atom_adder(bool is_aromatic) {
  return [is_aromatic](auto &ctx) {
    add_bracket_atom(ctx, x3::_attr(ctx), is_aromatic);
  };
}

constexpr auto set_chirality = [](auto &ctx) {
  const int idx = get_last_idx(ctx);
  x3::get<chirality_map_tag>(ctx).get()[idx] = x3::_attr(ctx);
  ABSL_DLOG(INFO) << "Setting chirality of atom " << idx << " to "
                  << static_cast<int>(x3::_attr(ctx));
};

constexpr auto set_atom_class = [](auto &) {
  ABSL_LOG(WARNING) << "Atom classes are currently not implemented";
};

const auto bracket_atom =  //
  x3::lit('[') >> ((-x3::uint_ >> element_symbol)[bracket_atom_adder(false)]
                   | (-x3::uint_ >> aromatic_symbol)[bracket_atom_adder(true)])
  >> -chirality[set_chirality] >> -hydrogen >> -charge
  >> -(x3::lit(':') >> x3::int_)[set_atom_class] >> x3::lit(']');

constexpr auto set_last_bond_data = [](auto &ctx) {
  x3::get<last_bond_data_tag>(ctx).get() = x3::_attr(ctx);
  ABSL_DLOG(INFO) << "Setting last bond data to " << x3::_attr(ctx);
};

constexpr auto set_last_bond_auto = [](auto &ctx) {
  x3::get<last_bond_data_tag>(ctx).get() = '\0';
  ABSL_DLOG(INFO) << "Setting last bond data to auto";
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
    ring_idx, {bond_data, current_idx}
  });

  // New ring index, nothing to do.
  if (is_new) {
    ABSL_DLOG(INFO) << "Adding ring index " << ring_idx
                    << ", src: " << current_idx << ", " << bond_data;
    return;
  }

  // Ring index already exists, add a bond.
  const RingData &data = it->second;

  char resolved_bond_data;
  if (data.bond_data == bond_data || bond_data == '\0') {
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
  const bool success =
    add_bond(mutator, data.atom_idx, current_idx, resolved_bond_data);
  if (ABSL_PREDICT_FALSE(!success)) {
    x3::_pass(ctx) = false;
    ABSL_LOG(WARNING) << "Failed to add ring bond from " << data.atom_idx
                      << " to " << current_idx;
    return;
  }

  if (resolved_bond_data == '/' || resolved_bond_data == '\\') {
    BondGeometryMap &bond_geometry_map = x3::get<bond_geometry_tag>(ctx);
    bond_geometry_map[std::make_pair(data.atom_idx, current_idx)] =
      resolved_bond_data;
  }

  map.erase(it);
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
  return [is_aromatic](auto &ctx) {
    add_atom(ctx, x3::_attr(ctx), -1, is_aromatic);
  };
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

void update_implicit_hydrogens(Molecule::Atom atom, AtomData &data) {
  // Required for correct bond order calculation
  data.set_implicit_hydrogens(0);

  int sum_bo = sum_bond_order(atom), normal_valence;

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

  data.set_implicit_hydrogens(std::max(0, normal_valence - sum_bo));
}

}  // namespace

Molecule read_smiles(std::string_view smiles) {
  Molecule mol;

  // Context variables
  MoleculeMutator mutator = mol.mutator();
  parser::AtomIdxStack stack;
  char last_bond_data = '.';
  parser::ChiralityMap chirality_map;
  parser::RingMap ring_map;
  parser::BondGeometryMap bond_geometry_map;

  stack.push(-1);

  auto parser = x3::with<parser::mutator_tag>(
    std::ref(mutator))[x3::with<parser::last_atom_stack_tag>(
    std::ref(stack))[x3::with<parser::last_bond_data_tag>(
    std::ref(last_bond_data))[x3::with<parser::chirality_map_tag>(
    std::ref(chirality_map))[x3::with<parser::ring_map_tag>(
    std::ref(ring_map))[x3::with<parser::bond_geometry_tag>(
    std::ref(bond_geometry_map))[parser::smiles]]]]]];

  auto begin = smiles.begin();
  bool success = x3::parse_main(begin, smiles.end(), parser, x3::unused);

  success = success && ring_map.empty();

  for (auto atom: mol) {
    if (atom.data().implicit_hydrogens() < 0) {
      update_implicit_hydrogens(atom, mutator.atom_data(atom.id()));
    }
  }

  if (success) {
    while (begin != smiles.end() && std::isspace(*begin) != 0) {
      ++begin;
    }
    mol.name() = std::string_view(&*begin, smiles.end() - begin);
  } else {
    ABSL_LOG(ERROR) << "Parsing failed: " << smiles;
    mol.clear();
  }

  return mol;
}
}  // namespace nuri
