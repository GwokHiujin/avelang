//===- AveLangTypes.cc - AveLang dialect type implementation ----------===//
//
// This file implements the types of the AveLang dialect.
//
//===----------------------------------------------------------------------===//

#include "AveLangTypes.h"
#include "AveLangDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "AveLangTypes.cpp.inc"

namespace causalflow::avelang::dialect {

void AveLangDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "AveLangTypes.cpp.inc"
        >();
}

// LayoutType custom parser - supports optional parameters
::mlir::Type LayoutType::parse(::mlir::AsmParser &parser) {
    ::llvm::SmallVector<int64_t> dims;
    ::llvm::SmallVector<int64_t> strides;

    // Try to parse less-than (optional parameters)
    if (parser.parseOptionalLess().succeeded()) {
        // Parse dims keyword
        if (parser.parseKeyword("dims") || parser.parseEqual() ||
            parser.parseLSquare()) {
            parser.emitError(parser.getCurrentLocation(),
                             "expected 'dims = [' in layout type");
            return ::mlir::Type();
        }

        // Parse dims array
        while (true) {
            int64_t dim;
            if (parser.parseOptionalInteger(dim).has_value()) {
                dims.push_back(dim);
                if (parser.parseOptionalComma().failed()) {
                    break;
                }
            } else {
                break;
            }
        }

        if (parser.parseRSquare()) {
            parser.emitError(parser.getCurrentLocation(),
                             "expected ']' after dims");
            return ::mlir::Type();
        }

        // Parse strides keyword
        if (parser.parseComma() || parser.parseKeyword("strides") ||
            parser.parseEqual() || parser.parseLSquare()) {
            parser.emitError(parser.getCurrentLocation(),
                             "expected ', strides = [' in layout type");
            return ::mlir::Type();
        }

        // Parse strides array
        while (true) {
            int64_t stride;
            if (parser.parseOptionalInteger(stride).has_value()) {
                strides.push_back(stride);
                if (parser.parseOptionalComma().failed()) {
                    break;
                }
            } else {
                break;
            }
        }

        if (parser.parseRSquare()) {
            parser.emitError(parser.getCurrentLocation(),
                             "expected ']' after strides");
            return ::mlir::Type();
        }

        if (parser.parseGreater()) {
            parser.emitError(parser.getCurrentLocation(),
                             "expected '>' in layout type");
            return ::mlir::Type();
        }
    }

    // Create LayoutType with parsed or empty parameters
    return LayoutType::get(parser.getContext(), dims, strides);
}

// LayoutType custom printer - only prints parameters if non-empty
void LayoutType::print(::mlir::AsmPrinter &printer) const {
    auto dims = getDims();
    auto strides = getStrides();

    // Only print parameters if there are any
    if (!dims.empty() || !strides.empty()) {
        printer << "<dims = [";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0)
                printer << ", ";
            printer << dims[i];
        }
        printer << "], strides = [";
        for (size_t i = 0; i < strides.size(); ++i) {
            if (i > 0)
                printer << ", ";
            printer << strides[i];
        }
        printer << "]>";
    }
}

// MemRefType custom parser - uses LayoutType to capture shape/stride
::mlir::Type MemRefType::parse(::mlir::AsmParser &parser) {
    ::mlir::Type layoutType;
    ::mlir::Type elementType;
    ::mlir::Attribute memorySpaceAttr;

    if (parser.parseLess()) {
        return ::mlir::Type();
    }

    if (parser.parseType(layoutType)) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected layout type in memref type");
        return ::mlir::Type();
    }
    if (!::mlir::isa<LayoutType>(layoutType)) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected ave.layout type in memref type");
        return ::mlir::Type();
    }

    if (parser.parseComma() || parser.parseType(elementType)) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected element type in memref type");
        return ::mlir::Type();
    }

    if (parser.parseOptionalComma().succeeded()) {
        if (parser.parseAttribute(memorySpaceAttr)) {
            parser.emitError(parser.getCurrentLocation(),
                             "expected memory space attribute in memref type");
            return ::mlir::Type();
        }
    }

    if (parser.parseGreater()) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected '>' in memref type");
        return ::mlir::Type();
    }

    return MemRefType::get(parser.getContext(), layoutType, elementType,
                           memorySpaceAttr);
}

// MemRefType custom printer - uses LayoutType to capture shape/stride
void MemRefType::print(::mlir::AsmPrinter &printer) const {
    printer << "<";
    auto memorySpaceAttr = getMemorySpace();
    printer << getLayout();
    printer << ", " << getElementType();
    if (memorySpaceAttr) {
        printer << ", " << memorySpaceAttr;
    }
    printer << ">";
}

bool MemRefType::hasStaticShape() const {
    for (auto dim : getShape()) {
        if (::mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return false;
        }
    }
    return true;
}

bool MemRefType::isDynamicDim(unsigned idx) const {
    auto shape = getShape();
    if (idx >= shape.size()) {
        return false;
    }
    return ::mlir::ShapedType::isDynamic(shape[idx]);
}

int64_t MemRefType::getDimSize(unsigned idx) const {
    auto shape = getShape();
    if (idx >= shape.size()) {
        return ::mlir::ShapedType::kDynamic;
    }
    return shape[idx];
}

int64_t MemRefType::getNumElements() const {
    int64_t numElements = 1;
    for (auto dim : getShape()) {
        if (::mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return ::mlir::ShapedType::kDynamic;
        }
        numElements *= dim;
    }
    return numElements;
}

// Type parsing hook - delegates to generated parser
::mlir::Type AveLangDialect::parseType(::mlir::DialectAsmParser &parser) const {
    ::llvm::StringRef mnemonic;
    ::mlir::Type type;

    if (generatedTypeParser(parser, &mnemonic, type).has_value()) {
        return type;
    }
    parser.emitError(parser.getCurrentLocation(), "unknown ave type: ")
        << mnemonic;
    return ::mlir::Type();
}

// Type printing hook - delegates to generated printer
void AveLangDialect::printType(::mlir::Type type,
                               ::mlir::DialectAsmPrinter &printer) const {
    if (failed(generatedTypePrinter(type, printer))) {
        printer << "<unknown>";
    }
}

} // namespace causalflow::avelang::dialect
