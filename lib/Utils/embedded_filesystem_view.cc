#include "embedded_filesystem_view.h"

namespace causalflow::avelang::utils {

EmbeddedFilesystemView &EmbeddedFilesystemView::getInstance() {
    static EmbeddedFilesystemView instance;
    return instance;
}

void EmbeddedFilesystemView::registerFile(const std::string &id,
                                          llvm::StringRef content) {
    files_[id] = content;
}

std::optional<llvm::StringRef>
EmbeddedFilesystemView::getFile(const std::string &id) const {
    auto it = files_.find(id);
    if (it != files_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool EmbeddedFilesystemView::hasFile(const std::string &id) const {
    return files_.count(id) > 0;
}

} // namespace causalflow::avelang::utils