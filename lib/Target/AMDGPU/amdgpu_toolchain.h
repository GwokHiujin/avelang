#pragma once

#include <llvm/Support/Program.h>
#include <llvm/TargetParser/Triple.h>
#include <memory>
#include <string>
#include <vector>

namespace causalflow::avelang::target::amdgpu {

/// Base class for tools in the AMDGPU toolchain
class Tool {
  public:
    Tool(const std::string &name, const std::string &shortName)
        : Name(name), ShortName(shortName) {}
    virtual ~Tool() = default;

    const std::string &getName() const { return Name; }
    const std::string &getShortName() const { return ShortName; }

  protected:
    std::string Name;
    std::string ShortName;
};

/// AMDGPU Linker tool (ld.lld)
class Linker : public Tool {
  public:
    Linker();

    /// Get the linker executable path
    std::string getLinkerPath() const;

    /// Construct linker command arguments for AMDGPU
    std::vector<std::string> constructLinkerArgs(
        const std::string &chipset, const std::vector<std::string> &deviceLibs,
        const std::string &inputFile, const std::string &outputFile) const;

  private:
    mutable std::string CachedLinkerPath;
    std::string findLinkerExecutable() const;
};

/// AMDGPU ToolChain - manages tools and their discovery
class AMDGPUToolChain {
  public:
    AMDGPUToolChain(const llvm::Triple &triple);
    ~AMDGPUToolChain();

    /// Get the target triple
    const llvm::Triple &getTriple() const { return Triple; }

    /// Get the linker tool
    Linker &getLinker() const;

    /// Find a program in the toolchain
    std::string getProgramPath(const std::string &name) const;

    /// Get default tool search paths
    std::vector<std::string> getToolSearchPaths() const;

  private:
    llvm::Triple Triple;
    mutable std::unique_ptr<Linker> LinkerTool;

    /// Common tool search paths
    std::vector<std::string> ToolSearchPaths;

    void initializeToolSearchPaths();
};

} // namespace causalflow::avelang::target::amdgpu