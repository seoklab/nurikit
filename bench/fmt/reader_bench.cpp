//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <absl/strings/str_cat.h>
#include <benchmark/benchmark.h>

#include "nuri/eigen_config.h"
#include "nuri/core/molecule.h"
#include "nuri/fmt/base.h"
#include "nuri/fmt/mol2.h"
#include "nuri/fmt/pdb.h"
#include "nuri/fmt/sdf.h"
#include "nuri/fmt/smiles.h"

namespace nuri {
namespace {
  constexpr std::string_view kProteinPath = "test/test_data/1ar1.pdb";
  constexpr int kSmallMolRepeat = 100;

  constexpr std::string_view kSmallMols[] = {
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H]"
    "(O)CC(=O)O",
    "CCCc1nn(C)c2c1nc(-c1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC)[nH]c2=O",
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "CN1CCC[C@H]1c1cccnc1",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
    "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "CN(C)C(=N)NC(=N)N",
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O",
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "[NH3+]CC(=O)[O-]",
    "C[N+](C)(C)CCO",
    "[O-]S(=O)(=O)[O-]",
    "c1ccc2ccccc2c1",
  };

  struct Corpus {
    std::string error;
    std::string pdb;
    std::string sdf_protein;
    std::string mol2_protein;
    std::string smi;
    std::string sdf_small;
    std::string mol2_small;
  };

  std::string slurp(std::string_view path) {
    std::ifstream ifs(std::string(path), std::ios::binary);
    return { std::istreambuf_iterator<char>(ifs),
             std::istreambuf_iterator<char>() };
  }

  Corpus build_corpus() {
    Corpus corpus;

    corpus.pdb = slurp(kProteinPath);
    if (corpus.pdb.empty()) {
      corpus.error = "cannot read protein input";
      return corpus;
    }

    Molecule protein;
    {
      std::istringstream is(corpus.pdb);
      PDBReader reader(is);
      auto result = reader.next()->parse();
      if (!result || result->data().empty()) {
        corpus.error = "cannot parse protein input";
        return corpus;
      }
      protein = std::move(result->data().front());
    }

    if (!write_sdf(corpus.sdf_protein, protein, 0)
        || !write_mol2(corpus.mol2_protein, protein, 0)) {
      corpus.error = "cannot write protein";
      return corpus;
    }

    std::string smi, sdf, mol2;
    for (std::string_view smiles: kSmallMols) {
      auto result = read_smiles(smiles);
      if (!result) {
        corpus.error = absl::StrCat("cannot parse small molecule: ", smiles);
        return corpus;
      }

      Molecule &mol = *result;
      mol.confs().push_back(Matrix3Xd::Random(3, mol.size()));

      if (!write_smiles(smi, mol) || !write_sdf(sdf, mol, 0, SDFVersion::kV2000)
          || !write_mol2(mol2, mol, 0)) {
        corpus.error = absl::StrCat("cannot write small molecule: ", smiles);
        return corpus;
      }
    }

    for (int i = 0; i < kSmallMolRepeat; ++i) {
      corpus.smi += smi;
      corpus.sdf_small += sdf;
      corpus.mol2_small += mol2;
    }

    return corpus;
  }

  const Corpus &corpus() {
    static const Corpus kCorpus = build_corpus();
    return kCorpus;
  }

  using CorpusField = std::string Corpus::*;

  template <bool kParse>
  void read_bench(benchmark::State &state, std::string_view fmt,
                  CorpusField field) {
    const Corpus &c = corpus();
    if (!c.error.empty()) {
      state.SkipWithError(c.error);
      return;
    }

    const std::string &data = c.*field;
    const MoleculeReaderFactory *factory =
        MoleculeReaderFactory::find_factory(fmt);

    std::istringstream is(data);
    int64_t records = 0, mols = 0, failures = 0;

    for (auto _: state) {
      is.clear();
      is.seekg(0);

      std::unique_ptr<MoleculeReader> reader = factory->from_stream(is);
      std::unique_ptr<MoleculeRecord> record = reader->make_record();
      while (reader->getnext(*record)) {
        ++records;
        if constexpr (kParse) {
          auto result = record->parse();
          benchmark::DoNotOptimize(result);
          if (result) {
            mols += static_cast<int64_t>(result->data().size());
          } else {
            ++failures;
          }
        }
      }
    }

    if (failures > 0) {
      state.SkipWithError("some records failed to parse");
      return;
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                            * static_cast<int64_t>(data.size()));
    state.counters["records"] =
        static_cast<double>(records) / static_cast<double>(state.iterations());
    if constexpr (kParse) {
      state.counters["mols"] =
          static_cast<double>(mols) / static_cast<double>(state.iterations());
    }
  }

#define NURI_READER_BENCH(name, fmt, field)                                    \
  BENCHMARK_CAPTURE(read_bench<false>, fill_##name, fmt, &Corpus::field)       \
      ->Unit(benchmark::kMillisecond);                                         \
  BENCHMARK_CAPTURE(read_bench<true>, parse_##name, fmt, &Corpus::field)       \
      ->Unit(benchmark::kMillisecond)

  NURI_READER_BENCH(pdb, "pdb", pdb);
  NURI_READER_BENCH(sdf_protein, "sdf", sdf_protein);
  NURI_READER_BENCH(mol2_protein, "mol2", mol2_protein);
  NURI_READER_BENCH(smi, "smi", smi);
  NURI_READER_BENCH(sdf_small, "sdf", sdf_small);
  NURI_READER_BENCH(mol2_small, "mol2", mol2_small);

#undef NURI_READER_BENCH
}  // namespace
}  // namespace nuri
