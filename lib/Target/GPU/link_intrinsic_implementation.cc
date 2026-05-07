#include "Utils/embedded_filesystem_view.h"
#include "Utils/monad_runner.h"
#include "gpu_passes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/MemoryBuffer.h>
#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>

namespace causalflow::avelang::target::gpu {

using namespace mlir;

namespace {

class LinkIntrinsicImplementationPass
    : public mlir::PassWrapper<LinkIntrinsicImplementationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        LinkIntrinsicImplementationPass)

    void runOnOperation() override;

  private:
    bool validateImpl(mlir::ModuleOp impl);
    bool loadLibrary(
        mlir::ModuleOp impl, mlir::ModuleOp module,
        llvm::StringMap<mlir::OwningOpRef<mlir::ModuleOp>> &libraryCache);
    bool loadEmbeddedLibrary(mlir::ModuleOp impl, mlir::ModuleOp module,
                             llvm::StringRef path,
                             mlir::OwningOpRef<mlir::ModuleOp> &library);
    bool getBytecode(mlir::ModuleOp impl, llvm::StringRef libraryName);
    bool parseEmbeddedBytecode(mlir::ModuleOp impl, llvm::StringRef libraryName,
                               mlir::ModuleOp module,
                               mlir::OwningOpRef<mlir::ModuleOp> &library);
    bool linkSymbols(
        mlir::ModuleOp impl, mlir::ModuleOp module,
        mlir::SymbolTable &symbolTable,
        llvm::StringMap<mlir::OwningOpRef<mlir::ModuleOp>> &libraryCache);
};

void LinkIntrinsicImplementationPass::runOnOperation() {
    using causalflow::avelang::MonadRunner;

    auto module = getOperation();
    llvm::SmallVector<mlir::ModuleOp> implModules;
    for (auto impl : module.getOps<mlir::ModuleOp>()) {
        if (impl->hasAttr("ave.intrinsic_impl")) {
            implModules.push_back(impl);
        }
    }

    if (implModules.empty())
        return;

    mlir::SymbolTable symbolTable(module);
    llvm::StringMap<mlir::OwningOpRef<mlir::ModuleOp>> libraryCache;

    for (auto impl : implModules) {
        MonadRunner<bool> runner(true);

        auto result =
            runner.Run([&]() { return validateImpl(impl); })
                .Run([&]() { return loadLibrary(impl, module, libraryCache); })
                .Run([&]() {
                    return linkSymbols(impl, module, symbolTable, libraryCache);
                })
                .code();

        if (!result) {
            signalPassFailure();
            return;
        }
    }
}

bool LinkIntrinsicImplementationPass::validateImpl(mlir::ModuleOp impl) {
    auto pathAttr =
        impl->getAttrOfType<mlir::StringAttr>("ave.intrinsic_library");
    if (!pathAttr) {
        impl.emitError() << "missing 'ave.intrinsic_library' attribute";
        return false;
    }
    return true;
}

bool LinkIntrinsicImplementationPass::loadLibrary(
    mlir::ModuleOp impl, mlir::ModuleOp module,
    llvm::StringMap<mlir::OwningOpRef<mlir::ModuleOp>> &libraryCache) {
    auto pathAttr =
        impl->getAttrOfType<mlir::StringAttr>("ave.intrinsic_library");
    auto path = pathAttr.getValue();
    auto &library = libraryCache[path];

    if (!library) {
        if (path.starts_with("embedded:")) {
            return loadEmbeddedLibrary(impl, module, path, library);
        }
        return false;
    }
    return true;
}

bool LinkIntrinsicImplementationPass::loadEmbeddedLibrary(
    mlir::ModuleOp impl, mlir::ModuleOp module, llvm::StringRef path,
    mlir::OwningOpRef<mlir::ModuleOp> &library) {
    using causalflow::avelang::MonadRunner;

    auto libraryName = path.substr(9); // Remove "embedded:" prefix

    MonadRunner<bool> runner(true);
    return runner.Run([&]() { return getBytecode(impl, libraryName); })
        .Run([&]() {
            return parseEmbeddedBytecode(impl, libraryName, module, library);
        })
        .code();
}

bool LinkIntrinsicImplementationPass::getBytecode(mlir::ModuleOp impl,
                                                  llvm::StringRef libraryName) {
    auto &registry =
        causalflow::avelang::utils::EmbeddedFilesystemView::getInstance();
    auto bytecode = registry.getFile(std::string(libraryName));

    if (!bytecode) {
        impl.emitError() << "embedded intrinsic library '" << libraryName
                         << "' not found in registry";
        return false;
    }
    return true;
}

bool LinkIntrinsicImplementationPass::parseEmbeddedBytecode(
    mlir::ModuleOp impl, llvm::StringRef libraryName, mlir::ModuleOp module,
    mlir::OwningOpRef<mlir::ModuleOp> &library) {
    auto &registry =
        causalflow::avelang::utils::EmbeddedFilesystemView::getInstance();
    auto bytecode = registry.getFile(std::string(libraryName));

    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        *bytecode, libraryName, /*RequiresNullTerminator=*/false);
    auto bufferRef = buffer->getMemBufferRef();

    if (!mlir::isBytecode(bufferRef)) {
        impl.emitError() << "embedded intrinsic library '" << libraryName
                         << "' is not MLIR bytecode";
        return false;
    }

    mlir::ParserConfig parserConfig(module.getContext());
    mlir::Block block;
    if (mlir::failed(mlir::readBytecodeFile(bufferRef, &block, parserConfig))) {
        impl.emitError() << "failed to parse embedded intrinsic bytecode '"
                         << libraryName << "'";
        return false;
    }

    mlir::ModuleOp parsedModule;
    for (mlir::Operation &op : block) {
        if (auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(&op)) {
            parsedModule = moduleOp;
            break;
        }
    }

    if (!parsedModule) {
        impl.emitError() << "embedded intrinsic bytecode '" << libraryName
                         << "' did not contain a module";
        return false;
    }

    library = llvm::cast<mlir::ModuleOp>(parsedModule->clone());
    return true;
}

bool LinkIntrinsicImplementationPass::linkSymbols(
    mlir::ModuleOp impl, mlir::ModuleOp module, mlir::SymbolTable &symbolTable,
    llvm::StringMap<mlir::OwningOpRef<mlir::ModuleOp>> &libraryCache) {
    auto pathAttr =
        impl->getAttrOfType<mlir::StringAttr>("ave.intrinsic_library");
    auto path = pathAttr.getValue();
    auto &library = libraryCache[path];

    auto libraryModule = library.get();
    if (!libraryModule) {
        impl.emitError() << "failed to load intrinsic library at " << path;
        return false;
    }

    for (auto &op : libraryModule.getOps()) {
        auto *clonedOp = op.clone();
        auto symbol = llvm::dyn_cast<mlir::SymbolOpInterface>(clonedOp);
        if (!symbol) {
            clonedOp->erase();
            impl.emitError() << "intrinsic library contains non-symbol op";
            return false;
        }

        if (auto existing = module.lookupSymbol(symbol.getName())) {
            if (auto existingFunc =
                    llvm::dyn_cast<mlir::func::FuncOp>(existing)) {
                if (auto newFunc =
                        llvm::dyn_cast<mlir::func::FuncOp>(clonedOp)) {
                    // Replace the existing function's body with the new
                    // implementation
                    existingFunc.getBody().getBlocks().clear();
                    mlir::IRMapping mapping;
                    newFunc.getBody().cloneInto(&existingFunc.getBody(),
                                                mapping);
                    // Copy attributes from the new function
                    for (auto namedAttr : newFunc->getAttrs()) {
                        existingFunc->setAttr(namedAttr.getName(),
                                              namedAttr.getValue());
                    }
                    clonedOp->erase();
                    continue;
                }
            }
            existing->erase();
        }

        symbol.setVisibility(mlir::SymbolTable::Visibility::Private);
        symbolTable.insert(symbol.getOperation());
    }

    impl.erase();
    return true;
};

} // namespace

std::unique_ptr<mlir::Pass> createLinkIntrinsicImplementationPass() {
    return std::make_unique<LinkIntrinsicImplementationPass>();
}

} // namespace causalflow::avelang::target::gpu