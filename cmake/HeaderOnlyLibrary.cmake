include(CMakeParseArguments)
# add_header_only_library(target build_include_dir install_include_dir [DEPENDS <targets>] [ALIAS <alias_name>])
function(add_header_only_library target_name build_interface install_interface)
  cmake_parse_arguments(_ARG "" "ALIAS" "DEPENDS" ${ARGN})
  add_library(${target_name} INTERFACE)
  if(_ARG_ALIAS)
    add_library(${_ARG_ALIAS} ALIAS ${target_name})
  endif()
  target_include_directories(${target_name} INTERFACE $<BUILD_INTERFACE:${build_interface}>
                                                      $<INSTALL_INTERFACE:${install_interface}>)
  target_link_libraries(${target_name} INTERFACE ${_ARG_DEPENDS})
endfunction(add_header_only_library)
