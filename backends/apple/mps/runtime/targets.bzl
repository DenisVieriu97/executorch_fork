#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    if runtime.is_oss:
        runtime.apple_library(
            name = "MPSBackend",
            srcs = [
              ("MPSExecutor.mm", [""]),
              ("MPSCompiler.mm", [""]),
              ("MPSBackend.mm",  [""]),
              ("MPSStream.mm",   [""]),
              ("MPSDevice.mm",   [""]),
            ],
            visibility = [
                "//executorch/examples/...",
                "@EXECUTORCH_CLIENTS",
            ],
            headers = [
              "MPSStream.h",
              "MPSDevice.h",
            ],
            compiler_flags = ["-std=c++17", "-iframeworkwithsysroot /System/Library/Frameworks/Foundation.framework"],
            external_deps = [
            "gflags",
            ],
            include_directories = ["/System/Library/Frameworks/Foundation.framework"],
            # precompiled_header =
            #   ("-F MetalPerformanceShaders",),
            exported_deps = [
                "//executorch/runtime/backend:interface",
            ],
            deps = [
                "//executorch/backends/apple/mps/utils:mps_utils",
                "//executorch/runtime/kernel:kernel_includes",
            ],
            exported_preprocessor_flags = ["-fobjc-arc", "-iframeworkwithsysroot /System/Library/Frameworks/Foundation.framework"],
            exported_header_style = "system",
            public_system_include_directories = ["/System/Library/Frameworks/Foundation.framework"],
            # define_static_target = True,

            preprocessor_flags = ['-framework Foundation'],
            platform_preprocessor_flags = ['-framework Foundation'],
            # linker_flags = ["-fobjc-arc", "-framework Metal"],
            link_whole = True,

            # platform_linker_flags = ["-fobjc-arc -F Foundation"],
        )
