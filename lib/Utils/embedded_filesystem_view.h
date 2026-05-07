#pragma once

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <optional>
#include <string>

namespace causalflow::avelang::utils {

/// Provides a centralized way to register and retrieve embedded files
/// identified by string keys (e.g., "nvvm_intrinsics.mlirbc", "shader.glsl")
/// Note: Does not own the file content - callers must ensure lifetime
class EmbeddedFilesystemView {
  public:
    static EmbeddedFilesystemView &getInstance();
    void registerFile(const std::string &id, llvm::StringRef content);
    std::optional<llvm::StringRef> getFile(const std::string &id) const;
    bool hasFile(const std::string &id) const;

  private:
    EmbeddedFilesystemView() = default;
    ~EmbeddedFilesystemView() = default;
    EmbeddedFilesystemView(const EmbeddedFilesystemView &) = delete;
    EmbeddedFilesystemView &operator=(const EmbeddedFilesystemView &) = delete;

    // Store file content as StringRefs (caller must ensure lifetime)
    llvm::StringMap<llvm::StringRef> files_;
};

} // namespace causalflow::avelang::utils