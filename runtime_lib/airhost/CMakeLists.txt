# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AIE_INCLUDE_DIRS}/../runtime_lib/x86_64/test_lib/include
    ${AIE_INCLUDE_DIRS}/../runtime_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../driver
)

add_definitions(-DLIBXAIENGINEV2)

# Only build the runtime if hsa was found and XAIE variables are set
if (hsa-runtime64_FOUND AND XILINX_XAIE_INCLUDE_DIR AND XILINX_XAIE_LIBS)
  get_filename_component(XILINX_XAIE_INCLUDE_PARENT
    ${XILINX_XAIE_INCLUDE_DIR} DIRECTORY)
  include_directories(${XILINX_XAIE_INCLUDE_DIR} ${XILINX_XAIE_INCLUDE_PARENT})
  find_library(ELF_LIBRARY elf REQUIRED)

  add_library(airhost STATIC
      memory.cpp
      queue.cpp
      runtime.cpp
      host.cpp
      pcie-ernic.cpp
      pcie-ernic-dev-mem-allocator.cpp
      network.cpp
  )
  set_property(TARGET airhost PROPERTY POSITION_INDEPENDENT_CODE ON)

  add_library(airhost_shared SHARED
      memory.cpp
      queue.cpp
      runtime.cpp
      host.cpp
      pcie-ernic.cpp
      pcie-ernic-dev-mem-allocator.cpp
      network.cpp
    )
  set_property(TARGET airhost_shared PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(airhost
    ${XILINX_XAIE_LIBS}
    dl
    ${ELF_LIBRARY}
    hsa-runtime64::hsa-runtime64
  )

  target_link_libraries(airhost_shared
    ${XILINX_XAIE_LIBS}
    hsa-runtime64::hsa-runtime64
    ${ELF_LIBRARY}
    dl
  )

  set_target_properties(airhost PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airhost)
  set_target_properties(airhost PROPERTIES
          ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airhost)
  install(TARGETS airhost DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIR_RUNTIME_TARGET}/airhost)

  set_target_properties(airhost_shared PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airhost)
  install(TARGETS airhost_shared DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIR_RUNTIME_TARGET}/airhost)

endif()

add_subdirectory(include)
