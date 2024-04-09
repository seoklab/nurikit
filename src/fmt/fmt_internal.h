//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FMT_FMT_INTERNAL_H_
#define NURI_FMT_FMT_INTERNAL_H_

#include <string>

#include <boost/spirit/home/x3.hpp>

namespace nuri {
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
