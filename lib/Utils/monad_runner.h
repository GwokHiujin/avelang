#pragma once

#include <functional>

namespace causalflow::avelang {
template <class T> class MonadRunner {
  public:
    MonadRunner(const T &okay) : okay_(okay), code_(okay) {}
    template <class Func, typename... Args>
    MonadRunner &Run(const Func &func, Args &&...args) {
        if (code_ == okay_) {
            if constexpr (sizeof...(args) == 0) {
                code_ = func();
            } else {
                code_ = std::invoke(func, std::forward<Args>(args)...);
            }
        }
        return *this;
    }
    T code() const { return code_; }

  private:
    const T okay_;
    T code_;
};
} // namespace causalflow::avelang
