#pragma once

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/Support/Allocator.h>

namespace causalflow::avelang::ast {

class ASTContext : public llvm::RefCountedBase<ASTContext> {
  public:
    llvm::BumpPtrAllocator &getAllocator() const { return BumpAlloc; }

    void *Allocate(size_t Size, unsigned Align = 8) const {
        return BumpAlloc.Allocate(Size, Align);
    }
    template <typename T> T *Allocate(size_t Num = 1) const {
        return static_cast<T *>(Allocate(Num * sizeof(T), alignof(T)));
    }
    void Deallocate(void *Ptr) const {}

  private:
    mutable llvm::BumpPtrAllocator BumpAlloc;
};

} // namespace causalflow::avelang::ast