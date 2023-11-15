#!/bin/bash
#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

set -exu

# shellcheck source=/dev/null
# source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME=$1
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

BUILD_TOOL=$2
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require cmake), exiting..."
  exit 1
fi

MPS_TESTS=$3
if [[ -z "${MPS_TESTS:-}" ]]; then
  echo "Missing flag specifying which tests to run, exiting..."
  exit 1
fi

which "${PYTHON_EXECUTABLE}"
CMAKE_OUTPUT_DIR=cmake-out
BUILD_TYPE=Debug

build_cmake_mps_executor_runner() {
  echo "Building mps_executor_runner"
  SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

  # Build and install executorch
  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && cmake -DBUCK2=buck2 \
          -DCMAKE_INSTALL_PREFIX=${CMAKE_OUTPUT_DIR} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DEXECUTORCH_BUILD_SDK=ON \
          -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
          -DEXECUTORCH_BUILD_MPS=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .)

  cmake --build ${CMAKE_OUTPUT_DIR} -j9 --target install --config ${BUILD_TYPE}
  CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"

  # Build the mps_executor_runner
  rm -rf cmake-out/examples/apple/mps
  cmake \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
      -Bcmake-out/examples/apple/mps \
      examples/apple/mps

  cmake --build cmake-out/examples/apple/mps -j9 --config ${BUILD_TYPE}
}

test_model_with_mps() {
  if [[ "${MODEL_NAME}" == "llama2" ]]; then
    cd examples/third-party/llama
    pip install -e .
    cd ../../..
  fi

  "${PYTHON_EXECUTABLE}" -m examples.apple.mps.scripts.mps_example --model_name="${MODEL_NAME}" --bundled

  OUTPUT_MODEL_PATH="${MODEL_NAME}_mps_bundled_fp16.pte"

  if [[ "${BUILD_TOOL}" == "cmake" ]]; then
    if [[ ! -f ${CMAKE_OUTPUT_DIR}/examples/apple/mps/mps_executor_runner ]]; then
      build_cmake_mps_executor_runner
    fi
    ./${CMAKE_OUTPUT_DIR}/examples/apple/mps/mps_executor_runner --model_path "${OUTPUT_MODEL_PATH}" --bundled_program
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only cmake is supported atm"
    exit 1
  fi
}

test_mps_model() {
  "${PYTHON_EXECUTABLE}" -m unittest "${MODEL_NAME}"

  TEST_NAME="${MODEL_NAME##*.}"
  OUTPUT_MODEL_PATH="${TEST_NAME}.pte"
  STR_LEN=${#OUTPUT_MODEL_PATH}
  FINAL_OUTPUT_MODEL_PATH=${OUTPUT_MODEL_PATH:5:$STR_LEN-5}

  if [[ "${BUILD_TOOL}" == "cmake" ]]; then
    if [[ ! -f ${CMAKE_OUTPUT_DIR}/examples/apple/mps/mps_executor_runner ]]; then
      build_cmake_mps_executor_runner
    fi
    ./${CMAKE_OUTPUT_DIR}/examples/apple/mps/mps_executor_runner --model_path "${FINAL_OUTPUT_MODEL_PATH}" --bundled_program
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only cmake is supported atm"
    exit 1
  fi
}

echo "Testing ${MODEL_NAME} with MPS..."
if [[ "${MPS_TESTS}" == false ]]; then
  test_model_with_mps
else
  test_mps_model
fi
