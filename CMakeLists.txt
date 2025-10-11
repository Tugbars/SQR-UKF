cmake_minimum_required(VERSION 3.16)
project(SQR_UKF C)

# ---- language / std ----
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ---- sources ----
set(SIMD_PUBLIC_INC ${CMAKE_CURRENT_SOURCE_DIR}/includes)

set(SIMD_SOURCES
    src/tran.c
    src/mul.c
    src/lup.c
    src/inv.c
    src/qr.c
    src/cholupdate.c
    src/sqr_ukf.c
)

# ---- library target ----
add_library(control_linalg_simd STATIC ${SIMD_SOURCES})
target_include_directories(control_linalg_simd PUBLIC ${SIMD_PUBLIC_INC})
set_target_properties(control_linalg_simd PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Warnings (library-only)
if (MSVC)
  target_compile_options(control_linalg_simd PRIVATE /W4)
else()
  target_compile_options(control_linalg_simd PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# AVX2/FMA for SIMD sources (library-only)
if (MSVC)
  target_compile_options(control_linalg_simd PRIVATE /arch:AVX2 /fp:fast)
else()
  target_compile_options(control_linalg_simd PRIVATE -mavx2 -mfma -O3 -ffast-math)
endif()

# Header feature switches used by the code
target_compile_definitions(control_linalg_simd PRIVATE
  LINALG_SIMD_ENABLE=1
  # TRAN_DISPATCH=1   # uncomment if you enabled runtime AVX checks in tran.c
)

# ---- test executable ----
add_executable(test_main main.c)
target_include_directories(test_main PRIVATE ${SIMD_PUBLIC_INC})
target_link_libraries(test_main PRIVATE control_linalg_simd)

# Warnings (test-only; no AVX2 flags)
if (MSVC)
  target_compile_options(test_main PRIVATE /W4)
else()
  target_compile_options(test_main PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "Building control_linalg_simd with AVX2/FMA enabled")
