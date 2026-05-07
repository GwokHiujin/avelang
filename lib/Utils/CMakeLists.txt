add_library(ave-lang-utils
  embedded_filesystem_view.cc
)

target_link_libraries(ave-lang-utils
  LLVMSupport
)