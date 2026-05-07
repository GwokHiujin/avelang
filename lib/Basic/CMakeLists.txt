add_library(ave-lang-basic
  diagnostic.cc
)

llvm_map_components_to_libnames(llvm_libs support)
target_link_libraries(ave-lang-basic 
  ${llvm_libs} 
  clangBasic
)