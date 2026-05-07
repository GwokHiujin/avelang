#include "avelang_parser_utils.h"
#include "avelang_parser.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace causalflow::avelang::frontend {

// ParsingContextGuard implementation
AveLangParser::ParsingContextGuard::ParsingContextGuard(
    AveLangParser *parser, const clang::SourceRange &range)
    : parent_(parser) {
    parent_->source_range_stack_.push_back(range);
}

AveLangParser::ParsingContextGuard::~ParsingContextGuard() {
    parent_->source_range_stack_.pop_back();
}

// MaybePyList implementation
MaybePyList::MaybePyList(const py::object &obj, const char *attr_name) {
    if (py::hasattr(obj, attr_name)) {
        try {
            list_ = obj.attr(attr_name).cast<py::list>();
            valid_ = true;
        } catch (const py::cast_error &) {
            valid_ = false;
        }
    }
}

const py::list &MaybePyList::GetEmptyList() {
    static const py::list empty_list;
    return empty_list;
}

// PythonInitializer implementation
PythonInitializer &PythonInitializer::GetInstance() {
    static PythonInitializer instance;
    return instance;
}

void PythonInitializer::EnsureInitialized() {}

PythonInitializer::PythonInitializer() {
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }
}

} // namespace causalflow::avelang::frontend