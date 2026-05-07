#include "rocm_installation_detector.h"
#include "avelang/config.h"

#include <algorithm>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <llvm/TargetParser/TargetParser.h>
#include <sstream>

namespace causalflow::avelang::target::amdgpu {

RocmInstallationDetector::RocmInstallationDetector() { detectDeviceLibrary(); }

void RocmInstallationDetector::scanLibDevicePath(llvm::StringRef Path) {
    const llvm::StringRef Suffix(".bc");
    const llvm::StringRef Suffix2(".amdgcn.bc");

    std::error_code EC;
    for (llvm::sys::fs::directory_iterator LI(Path, EC), LE; !EC && LI != LE;
         LI.increment(EC)) {
        llvm::StringRef FilePath = LI->path();
        llvm::StringRef FileName = llvm::sys::path::filename(FilePath);
        if (!FileName.ends_with(Suffix))
            continue;

        llvm::StringRef BaseName;
        if (FileName.ends_with(Suffix2))
            BaseName = FileName.drop_back(Suffix2.size());
        else if (FileName.ends_with(Suffix))
            BaseName = FileName.drop_back(Suffix.size());

        const llvm::StringRef ABIVersionPrefix = "oclc_abi_version_";
        if (BaseName == "ocml") {
            OCML = FilePath;
        } else if (BaseName == "ockl") {
            OCKL = FilePath;
        } else if (BaseName == "oclc_finite_only_off") {
            FiniteOnly.Off = FilePath;
        } else if (BaseName == "oclc_finite_only_on") {
            FiniteOnly.On = FilePath;
        } else if (BaseName == "oclc_daz_opt_on") {
            DenormalsAreZero.On = FilePath;
        } else if (BaseName == "oclc_daz_opt_off") {
            DenormalsAreZero.Off = FilePath;
        } else if (BaseName == "oclc_correctly_rounded_sqrt_on") {
            CorrectlyRoundedSqrt.On = FilePath;
        } else if (BaseName == "oclc_correctly_rounded_sqrt_off") {
            CorrectlyRoundedSqrt.Off = FilePath;
        } else if (BaseName == "oclc_unsafe_math_on") {
            UnsafeMath.On = FilePath;
        } else if (BaseName == "oclc_unsafe_math_off") {
            UnsafeMath.Off = FilePath;
        } else if (BaseName == "oclc_wavefrontsize64_on") {
            WavefrontSize64.On = FilePath;
        } else if (BaseName == "oclc_wavefrontsize64_off") {
            WavefrontSize64.Off = FilePath;
        } else if (BaseName.starts_with(ABIVersionPrefix)) {
            unsigned ABIVersionNumber;
            if (BaseName.drop_front(ABIVersionPrefix.size())
                    .getAsInteger(/*Redex=*/0, ABIVersionNumber))
                continue;
            ABIVersionMap[ABIVersionNumber] = FilePath.str();
        } else {
            // Process all bitcode filenames that look like
            // oclc_isa_version_XXX.amdgcn.bc
            const llvm::StringRef DeviceLibPrefix = "oclc_isa_version_";
            if (!BaseName.starts_with(DeviceLibPrefix))
                continue;

            llvm::StringRef IsaVersionNumber =
                BaseName.drop_front(DeviceLibPrefix.size());

            llvm::Twine GfxName = llvm::Twine("gfx") + IsaVersionNumber;
            llvm::SmallString<8> Tmp;
            LibDeviceMap.insert(
                std::make_pair(GfxName.toStringRef(Tmp), FilePath.str()));
        }
    }
}

std::vector<std::string>
RocmInstallationDetector::getInstallationPathCandidates() {
    return RocmPaths::getDeviceLibraryPaths();
}

void RocmInstallationDetector::detectDeviceLibrary() {
    auto candidates = getInstallationPathCandidates();

    for (const auto &path : candidates) {
        if (!llvm::sys::fs::exists(path))
            continue;

        scanLibDevicePath(path);

        if (allGenericLibsValid() && !LibDeviceMap.empty()) {
            LibDevicePath = path;
            HasDeviceLibrary = true;
            return;
        }

        // Clear partial results if this path didn't work
        OCML.clear();
        OCKL.clear();
        WavefrontSize64.On.clear();
        WavefrontSize64.Off.clear();
        FiniteOnly.On.clear();
        FiniteOnly.Off.clear();
        UnsafeMath.On.clear();
        UnsafeMath.Off.clear();
        DenormalsAreZero.On.clear();
        DenormalsAreZero.Off.clear();
        CorrectlyRoundedSqrt.On.clear();
        CorrectlyRoundedSqrt.Off.clear();
        ABIVersionMap.clear();
        LibDeviceMap.clear();
    }
}

bool RocmInstallationDetector::checkCommonBitcodeLibs(
    llvm::StringRef GPUArch, llvm::StringRef LibDeviceFile,
    DeviceLibABIVersion ABIVer) const {

    if (!hasDeviceLibrary()) {
        return false;
    }
    if (LibDeviceFile.empty()) {
        return false;
    }
    if (ABIVer.requiresLibrary() && getABIVersionPath(ABIVer).empty()) {
        return false;
    }
    return true;
}

std::vector<std::string> RocmInstallationDetector::getCommonBitcodeLibs(
    llvm::StringRef LibDeviceFile, bool Wave64, bool DAZ, bool FiniteOnly,
    bool UnsafeMathOpt, bool FastRelaxedMath, bool CorrectSqrt,
    DeviceLibABIVersion ABIVer) const {

    std::vector<std::string> BCLibs;

    auto AddBCLib = [&](llvm::StringRef BCFile) {
        if (!BCFile.empty()) {
            BCLibs.push_back(BCFile.str());
        }
    };

    AddBCLib(getOCMLPath());
    AddBCLib(getOCKLPath());
    AddBCLib(getDenormalsAreZeroPath(DAZ));
    AddBCLib(getUnsafeMathPath(UnsafeMathOpt || FastRelaxedMath));
    AddBCLib(getFiniteOnlyPath(FiniteOnly || FastRelaxedMath));
    AddBCLib(getCorrectlyRoundedSqrtPath(CorrectSqrt));
    AddBCLib(getWavefrontSize64Path(Wave64));
    AddBCLib(LibDeviceFile);

    auto ABIVerPath = getABIVersionPath(ABIVer);
    if (!ABIVerPath.empty())
        AddBCLib(ABIVerPath);

    return BCLibs;
}

//===----------------------------------------------------------------------===//
// RocmPaths Implementation (Config-based)
//===----------------------------------------------------------------------===//

namespace RocmPaths {

std::vector<std::string> getDeviceLibraryPaths() {
    std::vector<std::string> paths;

#ifdef WITH_AMDGPU
    // Primary path from CMake configuration (HIP package detection)
    paths.push_back(ROCM_DEVICE_LIBRARY_PATH);

    // Override with ROCM_PATH environment variable if set
    if (auto envPath = llvm::sys::Process::GetEnv("ROCM_PATH")) {
        paths.insert(paths.begin(), *envPath + "/amdgcn/bitcode");
    }
#endif

    return paths;
}

std::vector<std::string> getToolPaths() {
    std::vector<std::string> paths;

#ifdef WITH_AMDGPU
    // Parse semicolon-separated tool paths from CMake configuration
    std::string toolPathsStr = ROCM_TOOL_PATHS;
    std::stringstream ss(toolPathsStr);
    std::string path;

    while (std::getline(ss, path, ';')) {
        if (!path.empty()) {
            paths.push_back(path);
        }
    }

    // Override with ROCM_PATH environment variable if set
    if (auto envPath = llvm::sys::Process::GetEnv("ROCM_PATH")) {
        paths.insert(paths.begin(), *envPath + "/llvm/bin");
        paths.insert(paths.begin(), *envPath + "/bin");
    }
#endif

    return paths;
}

} // namespace RocmPaths

} // namespace causalflow::avelang::target::amdgpu