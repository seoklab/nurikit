//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <absl/base/call_once.h>
#include <absl/base/log_severity.h>
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>

namespace nuri_py {
namespace {
namespace py = pybind11;

int absl_severity_to_py_loglevel(absl::LogSeverity s) {
  switch (s) {
  case absl::LogSeverity::kFatal:
    return 50;
  case absl::LogSeverity::kError:
    return 40;
  case absl::LogSeverity::kWarning:
    return 30;
  case absl::LogSeverity::kInfo:
    return 20;
  }
  return 0;
}

class PyLogSink: public absl::LogSink {
public:
  /**
   * @pre The GIL must be held.
   */
  static void init() {
    static absl::once_flag flag;
    auto initializer = []() {
      absl::InitializeLog();

      PyLogSink::py_log_ =
        py::module_::import("logging").attr("getLogger")("nuri").attr("log");

      // Why no nolint for clang static analyzer?
#ifndef __clang_analyzer__
      // NOLINTNEXTLINE(*-owning-memory)
      absl::AddLogSink(new PyLogSink);
#endif
      absl::SetMinLogLevel(absl::LogSeverityAtLeast::kInfo);
      absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);

      ABSL_DLOG(INFO) << "initialized nurikit Python logging sink.";
    };
    absl::call_once(flag, initializer);
  }

  void Send(const absl::LogEntry &entry) override {
    const py::gil_scoped_acquire gil;

    py_log_(absl_severity_to_py_loglevel(entry.log_severity()),
            entry.text_message());
  }

private:
  // NOLINTNEXTLINE(*-identifier-naming,*-global-variables)
  static py::handle py_log_;
};

// NOLINTNEXTLINE(*-global-variables)
py::handle PyLogSink::py_log_;

PYBIND11_MODULE(_log_adapter, m) {
  m.def("_init", &PyLogSink::init);
}
}  // namespace
}  // namespace nuri_py
