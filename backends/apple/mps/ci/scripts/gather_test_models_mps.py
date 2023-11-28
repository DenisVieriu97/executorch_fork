#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import json
import os
from typing import Any

from examples.models import MODEL_NAME_TO_MODEL
from examples.xnnpack import MODEL_NAME_TO_OPTIONS
import unittest

MPS_TEST_SUITE_PATH = 'backends/apple/mps/test'
MPS_SUITE = unittest.defaultTestLoader.discover(MPS_TEST_SUITE_PATH)

BUILD_TOOLS = {
    "cmake": {"macos-14"},
}
DEFAULT_RUNNERS = {
    "macos-14": "macos-executorch",
}

def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Gather all models to test on CI for macOS MPS delegate")
    parser.add_argument(
        "--target-os",
        type=str,
        choices=["macos-14"],
        default="macos-14",
        help="the target OS",
    )
    return parser.parse_args()


def set_output(name: str, val: Any) -> None:
    """
    Set the GitHb output so that it can be accessed by other jobs
    """
    print(f"Setting {val} to GitHub output")

    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print(f"{name}={val}", file=env)
    else:
        print(f"::set-output name={name}::{val}")

def gather_mps_test_list(suite, mps_test_list):
    if hasattr(suite, '__iter__'):
        for x in suite:
          gather_mps_test_list(x, mps_test_list)
    else:
        mps_test_list.append(suite)

def gather_mps_tests(suite):
    mps_test_list = []
    gather_mps_test_list(suite, mps_test_list)
    return mps_test_list

def export_models_for_ci() -> None:
    """
    This gathers all the example models that we want to test on GitHub OSS CI
    """
    args = parse_args()
    target_os = args.target_os

    print(f"target os: {target_os}")

    # This is the JSON syntax for configuration matrix used by GitHub
    # https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
    models = {"include": []}
    for name in MODEL_NAME_TO_MODEL.keys():
        for build_tool in BUILD_TOOLS.keys():
            if target_os not in BUILD_TOOLS[build_tool]:
                continue

            record = {
                "build-tool": build_tool,
                "model": name,
                "runner": DEFAULT_RUNNERS.get(target_os),
            }

            models["include"].append(record)

    set_output("models", json.dumps(models))

    # Gather the list of all mps tests
    mps_models = {"include": []}
    mps_test_list = gather_mps_tests(MPS_SUITE)
    start_path = ".".join([MPS_TEST_SUITE_PATH.replace("/", "."), "test_mps"])
    for testcase in mps_test_list:
        if "test_mps" not in str(testcase.__class__):
          continue
        for build_tool in BUILD_TOOLS.keys():
            if target_os not in BUILD_TOOLS[build_tool]:
                continue

            print(f"method: {testcase._testMethodName}")
            print(f"{testcase.__class__}")
            print(f"{testcase.__class__.__name__}")
            cmd = ".".join([start_path, testcase.__class__.__name__, testcase._testMethodName])
            record = {
                "build-tool": build_tool,
                "model": cmd,
                "runner": DEFAULT_RUNNERS.get(target_os),
            }

            mps_models["include"].append(record)

    set_output("mps_models", json.dumps(mps_models))

if __name__ == "__main__":
    export_models_for_ci()
