//===- AveLangTypes.td - AveLang dialect types ----------------*- tablegen -*-===//
//
// This file defines the types of the AveLang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AVE_LANG_TYPES
#define AVE_LANG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "AveLangOps.td"

//===----------------------------------------------------------------------===//
// AveLang Type Definitions
//===----------------------------------------------------------------------===//

class AveLang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AveLang_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def LayoutType : AveLang_Type<"Layout", "layout"> {
    let summary = "Layout type for memory layout information";
    let description = [{
        The `ave.layout` type represents memory layout information
        that can be used in tensor operations. It contains dimensions and
        stride information for memory layout transformations.

        Example:
        ```mlir
        %layout = ave.make_layout %dims, %stride : none, none -> !ave.layout
        ```
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$dims,
        ArrayRefParameter<"int64_t">:$strides
    );

    let hasCustomAssemblyFormat = 1;
}

def MemRefType : AveLang_Type<"MemRef", "memref"> {
    let summary = "Unrestricted memref type";
    let description = [{
        The `!ave.memref` type mirrors MLIR's builtin `memref` type but
        relaxes validation to allow shapes, layouts, and memory spaces that
        may be invalid for MLIR's builtin memref. It uses a `!ave.layout`
        parameter to capture both shape and stride information prior to
        lowering.

        Example:
        ```mlir
        %layout = !ave.layout<dims = [4, 4], strides = [4, 1]>
        %buf : !ave.memref<!ave.layout, f32, #gpu.address_space<private>>
        ```
    }];

    let parameters = (ins
        "mlir::Type":$layout,
        "mlir::Type":$elementType,
        "mlir::Attribute":$memorySpace
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        LayoutType getLayoutType() const { return ::mlir::cast<LayoutType>(getLayout()); }
        llvm::ArrayRef<int64_t> getShape() const { return getLayoutType().getDims(); }
        llvm::ArrayRef<int64_t> getStrides() const { return getLayoutType().getStrides(); }
        int64_t getRank() const { return static_cast<int64_t>(getShape().size()); }
        bool hasRank() const { return true; }
        bool hasStaticShape() const;
        bool isDynamicDim(unsigned idx) const;
        int64_t getDimSize(unsigned idx) const;
        int64_t getNumElements() const;
    }];
}

#endif // AVE_LANG_TYPES
