# NuriKit Contributing Guidelines

You'd like to contribute to NuriKit? Great! We're happy to have you on board.

## Reporting issues

Please report any issues you find in the
[issue tracker](https://github.com/seoklab/nurikit/issues).

### Bug reports

If you've found a bug, please create an issue including the following
information in your report:

- The version or commit SHA of NuriKit you're using.
- The version of Python you're using (if it's related to the NuriKit python
  package).
- The OS and version you're using.
- A clear and concise description of what the bug is.
- A minimal code snippet that reproduces the bug.

Unfortunately, we can't fix bugs that we can't reproduce. If you're not sure
how to create a minimal code snippet, please check out
[this guide](https://stackoverflow.com/help/minimal-reproducible-example).

### Suggesting enhancements

If you have an idea for a new feature, please create a new issue with the
following information:

- Rationale for the new feature.
- Journal references for the new feature (if it is scientific).
- Tools that already implement the new feature (if any).

## Developer guide

### Setting up the development environment

Basically, you will be OK with the default packages provided by Ubuntu 20.04+.
The specific requirements for building NuriKit are:

- Linux (macOS builds in CI, but is untested as a development platform)
- [Git](https://git-scm.com/)
- [CMake](https://cmake.org/), version 3.16+
- C/C++ Compilers
  - [GCC](https://gcc.gnu.org/), version 9+
  - [Clang](https://clang.llvm.org/), version 10+

Also, you should have
[installed the latest version of pre-commit](https://pre-commit.com/#install)
and run `pre-commit install --install-hooks` in the root of the repository.

clang-format and clang-tidy are used to format and lint the source code. You
should run these tools before submitting the PR. You can run the tools with
the provided script: `./scripts/run_clang_tools.sh`:

```shellscript
# All files
./scripts/run_clang_tools.sh [-j<nproc>]
# Specific files
./scripts/run_clang_tools.sh [-j<nproc>] src/core/element.cpp include/nuri/core/element.h
```

The script assumes version 16+ of both tools. CI pins the version to a specific
major release; we recommend the same version for local development to avoid
unnecessary churn. Refer
[.github/workflows/_run-clang-tools.yaml](.github/workflows/_run-clang-tools.yaml)
for the exact version used in CI.

### Commit messages

We recommend the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) style
for commit messages. This will help us to automatically generate
changelogs and release notes. We've setup the pre-commit hook to check the
commit messages, so it will complain if you don't follow the convention. If
you're not sure how to write a commit message, please check out
[this guide](https://www.conventionalcommits.org/en/v1.0.0/#summary).

### Code style

#### C++

All code written in C++ should follow our C++ style guide. It is based on the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
However, some of the rules are tailored to our needs. These are the rules that
are different from the Google C++ Style Guide:

- [Static storage duration](http://en.cppreference.com/w/cpp/language/storage_duration#Storage_duration) objects that are not
  [trivially destructible](http://en.cppreference.com/w/cpp/types/is_destructible)
  are allowed, but they should be declared `const` and must not reference other
  global variables that are not in the same translation unit. This includes
  references to other global variables in the constructor of the global
  variable. This is to avoid the
  [static initialization order fiasco](https://isocpp.org/wiki/faq/ctors#static-init-order).
  Prefer `constexpr` variables over `const` variables if possible.
- The use of `explicit` keyword for single-argument constructors is optional.
- We allow exceptions inside *internal functions*[^1], but they must not escape
  the public C++ API.
- All functions, including member functions, must follow the naming convention
  of `lower_case`. This is to match the Python naming convention.
- For source files, `.cpp` extensions must be used.
- Code should be formatted using `clang-format`, with the provided
  `.clang-format` file. If part of the code needs to be formatted manually for
  some reasons, `// clang-format off` and `// clang-format on` comments may be
  used.

[^1]: Internal functions are functions that are not part of the public C++
      API. They are not declared in the header files, or they are declared
      inside the `nuri::internal` namespace.

The remaining rules are the same as the Google C++ Style Guide.

#### Python

All code written in Python should follow the
[PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.

### Pull requests

Once you've finished working on an issue, please create a new pull request with
the following required information:

- Code review checklist (will be automatically generated from the
  [PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) file).
- Tracking issue(s).
- A clear and concise description of what the pull request does.
