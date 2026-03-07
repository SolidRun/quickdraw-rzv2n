# Quick Draw — Renesas RZ/V2N
#   app_quickdraw     — C++ DRP-AI inference server (Unix socket, for Python GUI)
#   app_quickdraw_gui — C++ GTK3 GUI + DRP-AI (single binary, no socket)
#
# Build (inside Docker container):
#   mkdir -p build && cd build
#   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain/runtime.cmake ..
#   make -j$(nproc)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(quickdraw CXX)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

# TVM_ROOT = DRP-AI TVM repository root ($TVM_ROOT env var)
set(TVM_ROOT "$ENV{TVM_ROOT}")
if(NOT TVM_ROOT)
    message(FATAL_ERROR "TVM_ROOT env var not set. Export TVM_ROOT=/drp-ai_tvm")
endif()

# TVM/MERA2 includes
include_directories(
    ${TVM_ROOT}/tvm/include
    ${TVM_ROOT}/setup/include
    ${TVM_ROOT}/tvm/3rdparty/dlpack/include
    ${TVM_ROOT}/tvm/3rdparty/dmlc-core/include
    ${TVM_ROOT}/tvm/3rdparty/compiler-rt
    ${TVM_ROOT}/apps
    ${TVM_ROOT}/apps/include
    ${TVM_ROOT}/3rdparty/spdlog/include
    ${TVM_ROOT}/3rdparty/asio/asio/include
    ${TVM_ROOT}/how-to/sample_app_v2h/common_files
    ${TVM_ROOT}/how-to/sample_app_v2h/common_files/package
)

# MERA2 runtime libraries (V2N uses v2h runtime)
set(LIBMERA_RT_PATH ${TVM_ROOT}/obj/build_runtime/v2h/lib)

add_definitions(-DMERA_DRP_RUNTIME)

# ── DRP-AI Inference Server ──
add_executable(app_quickdraw
    src/main.cpp
    src/drpai_inference.cpp
    src/classification.cpp
    src/preprocessing.cpp
    ${TVM_ROOT}/apps/MeraDrpRuntimeWrapper.cpp
    ${TVM_ROOT}/apps/PreRuntimeV2H.cpp
)

target_include_directories(app_quickdraw PRIVATE
    src
    ${TVM_ROOT}/apps
    ${TVM_ROOT}/apps/include
)

target_compile_options(app_quickdraw PRIVATE -O3 -mtune=cortex-a55 -Wall)
target_compile_definitions(app_quickdraw PRIVATE V2H V2N KDLDRPAI)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,${LIBMERA_RT_PATH} -Wl,-rpath-link,${LIBMERA_RT_PATH}")

target_link_directories(app_quickdraw PRIVATE ${LIBMERA_RT_PATH})
target_link_libraries(app_quickdraw PRIVATE
    mera2_runtime
    mera2_plan_io
    drp_tvm_rt
    pthread
)

message(STATUS "TVM_ROOT: ${TVM_ROOT}")
message(STATUS "MERA2 runtime: ${LIBMERA_RT_PATH}")

# ── C++ GTK3 GUI + DRP-AI (single binary) ──
# GTK3 headers from the Yocto SDK sysroot
set(GTK3_SYSROOT "${CMAKE_SYSROOT}")
set(GTK3_INCLUDE_DIRS
    ${GTK3_SYSROOT}/usr/include/gtk-3.0
    ${GTK3_SYSROOT}/usr/include/glib-2.0
    ${GTK3_SYSROOT}/usr/lib/glib-2.0/include
    ${GTK3_SYSROOT}/usr/lib64/glib-2.0/include
    ${GTK3_SYSROOT}/usr/include/pango-1.0
    ${GTK3_SYSROOT}/usr/include/harfbuzz
    ${GTK3_SYSROOT}/usr/include/cairo
    ${GTK3_SYSROOT}/usr/include/gdk-pixbuf-2.0
    ${GTK3_SYSROOT}/usr/include/atk-1.0
    ${GTK3_SYSROOT}/usr/include/pixman-1
    ${GTK3_SYSROOT}/usr/include/freetype2
    ${GTK3_SYSROOT}/usr/include/libpng16
    ${GTK3_SYSROOT}/usr/include/libdrm
)

add_executable(app_quickdraw_gui
    src/main_gui.cpp
    src/gui.cpp
    src/drpai_inference.cpp
    src/classification.cpp
    src/preprocessing.cpp
    ${TVM_ROOT}/apps/MeraDrpRuntimeWrapper.cpp
    ${TVM_ROOT}/apps/PreRuntimeV2H.cpp
)

target_include_directories(app_quickdraw_gui PRIVATE
    src
    ${TVM_ROOT}/apps
    ${TVM_ROOT}/apps/include
    ${GTK3_INCLUDE_DIRS}
)

target_compile_options(app_quickdraw_gui PRIVATE -O3 -mtune=cortex-a55 -Wall)
target_compile_definitions(app_quickdraw_gui PRIVATE V2H V2N KDLDRPAI)
target_link_directories(app_quickdraw_gui PRIVATE ${LIBMERA_RT_PATH})
target_link_libraries(app_quickdraw_gui PRIVATE
    mera2_runtime
    mera2_plan_io
    drp_tvm_rt
    pthread
    gtk-3
    gdk-3
    cairo
    glib-2.0
    gobject-2.0
    pango-1.0
    pangocairo-1.0
    gdk_pixbuf-2.0
    atk-1.0
    gio-2.0
)

install(TARGETS app_quickdraw app_quickdraw_gui RUNTIME DESTINATION bin)
