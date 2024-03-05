//
// Project nurikit - Copyright 2023 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <absl/base/call_once.h>
#include <absl/base/internal/raw_logging.h>
#include <absl/base/log_severity.h>
#include <absl/log/absl_check.h>
#include <absl/log/absl_log.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log_entry.h>
#include <absl/log/log_sink.h>

#include "absl/log/log.h"
#ifndef __clang_analyzer__
#include <absl/log/log_sink_registry.h>
#endif

namespace nuri {
namespace python_internal {
namespace {
int absl_severity_to_py_loglevel(absl::LogSeverity s) {
  switch (absl::NormalizeLogSeverity(s)) {
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

absl::LogSeverityAtLeast py_loglevel_to_absl_severity(int level) {
  if (level <= 0)
    return absl::LogSeverityAtLeast::kWarning;

  if (level >= 50)
    return absl::LogSeverityAtLeast::kFatal;
  if (level >= 40)
    return absl::LogSeverityAtLeast::kError;
  if (level >= 30)
    return absl::LogSeverityAtLeast::kWarning;

  return absl::LogSeverityAtLeast::kInfo;
}

int message_loglevel(const absl::LogEntry &entry) {
  // verbose logging -> debug level
  if (entry.verbosity() != absl::LogEntry::kNoVerbosityLevel)
    return 10;
  return absl_severity_to_py_loglevel(entry.log_severity());
}

void set_log_level(int level) {
  absl::LogSeverityAtLeast severity = py_loglevel_to_absl_severity(level);
  absl::SetMinLogLevel(severity);
  ABSL_DLOG(INFO) << "Setting log level " << severity;

  if (level > 0 && level <= 10) {
    int prev = absl::SetGlobalVLogLevel(3);
    ABSL_DVLOG(1) << "Setting verbose log level " << prev << " -> 3";
  } else {
    int prev = absl::SetGlobalVLogLevel(0);
    ABSL_DLOG(INFO) << "Setting verbose log level " << prev << " -> 0";
  }
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

      logger_ = py::module_::import("logging").attr("getLogger")("nuri");
      logger_.inc_ref();

      // Why no nolint for clang static analyzer?
#ifndef __clang_analyzer__
      // NOLINTNEXTLINE(*-owning-memory)
      absl::AddLogSink(new PyLogSink);
#endif
      absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);
      set_log_level(10);
      ABSL_DLOG(INFO) << "initialized nurikit Python logging sink.";
    };

    absl::call_once(flag, initializer);
  }

  void Send(const absl::LogEntry &entry) override try {
    const py::gil_scoped_acquire gil;

    try {
      logger_.attr("log")(message_loglevel(entry), entry.text_message());
    } catch (py::error_already_set &e) {
      e.discard_as_unraisable("nurikit internal logging");
    }
  } catch (...) {
    ABSL_RAW_LOG(ERROR, "unknown error while logging (original message: %s)",
                 entry.text_message_with_prefix_and_newline_c_str());
  }

private:
  // NOLINTNEXTLINE(*-identifier-naming,*-global-variables)
  static py::handle logger_;
};

// NOLINTNEXTLINE(*-global-variables)
py::handle PyLogSink::logger_;

NURI_PYTHON_MODULE(m) {
  m.def("_init", &PyLogSink::init);
  m.def("set_log_level", set_log_level);
}
}  // namespace
}  // namespace python_internal
}  // namespace nuri
