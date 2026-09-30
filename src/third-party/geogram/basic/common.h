/*
 *  Copyright (c) 2000-2022 Inria
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Contact: Bruno Levy
 *
 *     https://www.inria.fr/fr/bruno-levy
 *
 *     Inria,
 *     Domaine de Voluceau,
 *     78150 Le Chesnay - Rocquencourt
 *     FRANCE
 *
 */

#ifndef GEOGRAM_BASIC_COMMON
#define GEOGRAM_BASIC_COMMON

#include <geogram/api/defs.h>

// iostream should be included before anything else,
// otherwise 'cin', 'cout' and 'cerr' will be uninitialized.
#include <iostream>

/**
 * \file geogram/basic/common.h
 * \brief Common include file, providing basic definitions. Should be
 *  included before anything else by all header files in Vorpaline.
 */


/**
 * \brief Global Vorpaline namespace
 * \details This namespace contains all the Vorpaline classes and functions
 * organized in sub-namespaces.
 */
/**
 * \def GEO_OS_LINUX
 * \brief This macro is set on Linux systems.
 *
 * \def GEO_OS_UNIX
 * \brief This macro is set on Unix systems.
 *
 * \def GEO_OS_APPLE
 * \brief This macro is set on Apple systems.
 *
 * \def GEO_COMPILER_GCC
 * \brief This macro is set if the source code is compiled with GNU's gcc.
 *
 * \def GEO_COMPILER_CLANG
 * \brief This macro is set if the source code is compiled with clang.
 */

#if defined(__linux__)
#define GEO_OS_LINUX
#define GEO_OS_UNIX
#elif defined(__APPLE__)
#define GEO_OS_APPLE
#define GEO_OS_UNIX
#else
#error "Unsupported operating system"
#endif

#if defined(__clang__)
#define GEO_COMPILER_CLANG
#elif defined(__GNUC__)
#define GEO_COMPILER_GCC
#else
#error "Unsupported compiler"
#endif
#define GEO_COMPILER_GCC_FAMILY

// Silence warnings for alloca()
// We use it at different places to allocate objects on the stack
// (for instance, in multi-precision predicates).
#ifdef GEO_COMPILER_CLANG
#pragma GCC diagnostic ignored "-Walloca"
#endif

// =============================== Parallel STL ============================

// For now, deactivate parallel STL if in a Pluggable Softare Module
// (because if compiling with gcc, this forces linking tbb which may
//  be not suitable)


#endif
