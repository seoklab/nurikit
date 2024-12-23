//
// Project NuriKit - Copyright 2024 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NURI_FUZZ_FUZZ_UTILS_H_
#define NURI_FUZZ_FUZZ_UTILS_H_

#define NURI_FUZZ_MAIN(data, size)                                             \
  extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)

NURI_FUZZ_MAIN(/* data */, /* size */);

#endif /* NURI_FUZZ_FUZZ_UTILS_H_ */
