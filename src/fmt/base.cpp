//
// Project NuriKit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include "nuri/fmt/base.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <ios>
#include <istream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/algorithm/container.h>
#include <absl/base/optimization.h>
#include <absl/container/flat_hash_map.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/strings/ascii.h>
#include <absl/strings/charset.h>

#include "nuri/utils.h"

namespace nuri {
namespace {
absl::flat_hash_map<std::string, const MoleculeReaderFactory *> &
reader_factory_registry() {
  static absl::flat_hash_map<std::string, const MoleculeReaderFactory *> ret;
  return ret;
}
}  // namespace

namespace internal {
std::string ascii_safe(std::string_view str) {
  std::string ret(str);

  auto begin = absl::c_find_if_not(ret, absl::ascii_isspace),
       end = std::find_if_not(ret.rbegin(), ret.rend(), absl::ascii_isspace)
                 .base();

  if (ABSL_PREDICT_FALSE(begin == ret.end())) {
    ret.clear();
    return ret;
  }

  std::fill(ret.begin(), begin, ' ');

  for (auto it = begin; it < end; ++it) {
    char &c = *it;
    if (absl::ascii_isspace(c)) {
      c = '_';
    } else if (ABSL_PREDICT_FALSE(!absl::ascii_isprint(c))) {
      c = '?';
    }
  }

  std::fill(end, ret.end(), ' ');

  return ret;
}

constexpr absl::CharSet kNewlines("\f\n\r");

std::string ascii_newline_safe(std::string_view str) {
  std::string ret(str);

  for (char &c: ret) {
    if (kNewlines.contains(c)) {
      c = ' ';
    } else if (!absl::ascii_isspace(c)
               && ABSL_PREDICT_FALSE(!absl::ascii_isprint(c))) {
      c = '?';
    }
  }

  return ret;
}
}  // namespace internal

const MoleculeReaderFactory *
MoleculeReaderFactory::find_factory(std::string_view name) {
  const absl::flat_hash_map<std::string, const MoleculeReaderFactory *> &reg =
      reader_factory_registry();

  auto it = reg.find(name);
  if (it == reg.end()) {
    return nullptr;
  }
  return it->second;
}

bool MoleculeReaderFactory::register_factory(
    std::unique_ptr<MoleculeReaderFactory> factory,
    const std::vector<std::string> &names) {
  static std::vector<std::unique_ptr<MoleculeReaderFactory>> factories;

  MoleculeReaderFactory *f = factories.emplace_back(std::move(factory)).get();
  // GCOV_EXCL_START
  ABSL_LOG_IF(WARNING, names.empty()) << "Empty name list for factory";
  // GCOV_EXCL_STOP

  for (const auto &name: names) {
    register_for_name(f, name);
  }

  return true;
}

void MoleculeReaderFactory::register_for_name(
    const MoleculeReaderFactory *factory, std::string_view name) {
  auto [_, inserted] =
      reader_factory_registry().insert_or_assign(name, factory);
  // GCOV_EXCL_START
  ABSL_LOG_IF(WARNING, !inserted)
      << "Duplicate factory name: " << name
      << ". Overwriting existing factory (is this intended?).";
  // GCOV_EXCL_STOP
}

namespace {
const char *find_after_last_delim(const char *begin, const char *end,
                                  char delim) {
  // after_last points to the *next* character of last delim in the buffer,
  // due to the nature of std::reverse_iterator.
  return std::find(std::make_reverse_iterator(end),
                   std::make_reverse_iterator(begin), delim)
      .base();
}
}  // namespace

void ReversedStream::reset() {
  is_->clear();
  is_->seekg(0, std::ios::end);
  if (is_->tellg() == 0) {
    is_->setstate(std::ios::eofbit);
  }

  read_block();

  if (prev_ > 0 && buf_[prev_ - 1] == delim_) {
    --prev_;
  }
}

bool ReversedStream::getline(std::string &line) {
  line.clear();

  if (!*is_) {
    return false;
  }

  do {
    auto after_last =
        find_after_last_delim(buf_.cbegin(), buf_.cbegin() + prev_, delim_);
    line.insert(line.begin(), after_last, buf_.cbegin() + prev_);

    // Found, done
    if (after_last > buf_.cbegin()) {
      prev_ = after_last - buf_.cbegin() - 1;
      break;
    }

    read_block();
  } while (*is_);

  return true;
}

void ReversedStream::read_block() {
  if (is_->eof()) {
    is_->setstate(std::ios::failbit);
    prev_ = 0;
    return;
  }

  const size_t unread = is_->tellg();
  size_t read_size = nuri::min(unread, buf_.size());
  std::ios::off_type offset = -static_cast<std::ios::off_type>(read_size);

  is_->seekg(offset, std::ios::cur);
  is_->read(buf_.data(), static_cast<std::streamsize>(read_size));
  prev_ = is_->gcount();
  ABSL_DCHECK(!*is_ || is_->gcount() == read_size);

  is_->seekg(offset, std::ios::cur);
  if (is_->tellg() == 0) {
    is_->setstate(std::ios::eofbit);
  }
}
}  // namespace nuri
