#pragma once

#include <clang/Basic/SourceManager.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace pybind11 {
class object;
}

namespace causalflow::avelang::frontend {

// Utility class for handling optional Python lists with range-based for loop
// support
class __attribute__((visibility("hidden"))) MaybePyList {
  public:
    MaybePyList(const pybind11::object &obj, const char *attr_name);

    bool IsValid() const { return valid_; }

    // Enable range-based for loop: for (auto item : maybe_list)
    auto begin() const {
        return valid_ ? list_.begin() : GetEmptyList().begin();
    }
    auto end() const { return valid_ ? list_.end() : GetEmptyList().end(); }

  private:
    static const pybind11::list &GetEmptyList();

    pybind11::list list_;
    bool valid_ = false;
};

// Singleton for Python interpreter initialization
class PythonInitializer {
  public:
    static PythonInitializer &GetInstance();
    void EnsureInitialized();

  private:
    PythonInitializer();
    ~PythonInitializer() = default;

    // Delete copy constructor and assignment operator
    PythonInitializer(const PythonInitializer &) = delete;
    PythonInitializer &operator=(const PythonInitializer &) = delete;
};

} // namespace causalflow::avelang::frontend