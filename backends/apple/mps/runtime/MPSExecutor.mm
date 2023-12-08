//
//  Copyright (c) 2023 Apple Inc. All rights reserved.
//  Provided subject to the LICENSE file in the top level directory.
//

#define EXIR_MPS_DELEGATE 1

#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/backends/apple/mps/runtime/MPSStream.h>
#include <executorch/backends/apple/mps/utils/OperationUtils.h>

#include "MPSExecutor.h"

@interface MPSNDArray ()
-(nonnull instancetype) initWithBuffer:(id<MTLBuffer> _Nonnull) buffer
                            descriptor:(MPSNDArrayDescriptor * _Nonnull) descriptor;
@end

@interface MPSNDArrayDescriptor ()
@property (readwrite, nonatomic) BOOL preferPackedRows;
@property (readwrite, nonatomic)  NSUInteger rowBytes;
@end


namespace torch {
namespace executor {
namespace mps {
namespace delegate {

__ET_NODISCARD Error
MPSExecutor::set_inputs_outputs(std::vector<const Tensor*>& inputs, std::vector<const Tensor*>& outputs) {
  ET_CHECK_OR_RETURN_ERROR(inputs.size() == getNumInputs(), Internal, "Inputs mismatch");
  ET_CHECK_OR_RETURN_ERROR(outputs.size() == getNumOutputs(), Internal, "Outputs mismatch");

#if !TARGET_OS_SIMULATOR
  if (outputsArray_ != nil) {
    return Error::Ok;
  }
#endif

  inputsArray_ = [[NSMutableArray<MPSGraphTensorData *> alloc] init];
  outputsArray_ = [[NSMutableArray<MPSGraphTensorData *> alloc] init];

  auto calculateRowBytes = [] (MPSNDArrayDescriptor *desc) {
    auto rowSize = [desc lengthOfDimension:0];
    auto byteTypes = MPSSizeofMPSDataType([desc dataType]);
    return ((rowSize * byteTypes + 63) / 64) * 64;
  };

  for (int i = 0; i < inputs.size(); i++) {
    MPSNDArrayDescriptor *tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:[inputShapes_[i] dataType]
                                                                              shape:[inputShapes_[i] shape]];
    tensorDesc.rowBytes = calculateRowBytes(tensorDesc);
    MPSNDArray *ndArrayData = [[MPSNDArray alloc] initWithDevice:MPSDevice::getInstance()->device()
                                                      descriptor:tensorDesc];
    MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMPSNDArray:ndArrayData];
    [inputsArray_ addObject:tensorData];
  }

  for (int i = 0; i < outputs.size(); i++) {
    MPSNDArrayDescriptor *tensorDesc = [MPSNDArrayDescriptor descriptorWithDataType:[outputShapes_[i] dataType]
                                                                              shape:[outputShapes_[i] shape]];
    tensorDesc.rowBytes = calculateRowBytes(tensorDesc);
    MPSNDArray *ndArrayData = [[MPSNDArray alloc] initWithDevice:MPSDevice::getInstance()->device()
                                                      descriptor:tensorDesc];
    MPSGraphTensorData* tensorData = [[MPSGraphTensorData alloc] initWithMPSNDArray:ndArrayData];
    [outputsArray_ addObject:tensorData];
  }

  return Error::Ok;
}

__ET_NODISCARD Error MPSExecutor::forward(std::vector<const Tensor*>& inputs,
                                          std::vector<const Tensor*>& outputs) {
  Error err = Error::Ok;
  MPSStream* mpsStream = getDefaultMPSStream();
  if (!mpsStream->commitAndContinueEnabled() && !mpsStream->hasLivecommandBuffer()) {
    for (int i = 0; i < outputs.size(); i++)
      [inputsArray_[i].mpsndarray writeBytes:inputs[i]->mutable_data_ptr<uint8_t>()
                                 strideBytes:nil];

    [executable_ runWithMTLCommandQueue:mpsStream->commandQueue()
                            inputsArray:inputsArray_
                           resultsArray:outputsArray_
                    executionDescriptor:mpsStream->getExecutableExecutionDescriptor()];
  } else {
    // TODO input copies.
    exit(1);
    id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
    [executable_ encodeToCommandBuffer:commandBuffer
                          inputsArray:inputsArray_
                          resultsArray:outputsArray_
                  executionDescriptor:mpsStream->getExecutableExecutionDescriptor()];
  }

  if (mpsStream->commitAndContinueEnabled()) {
    // TODO command buffer completion handler needs to copy the data.
    exit(1);
    err = mpsStream->synchronize(SyncType::COMMIT_AND_CONTINUE);
  } else {
    err = mpsStream->synchronize(SyncType::COMMIT_AND_WAIT);
    for (int i = 0; i < outputs.size(); i++)
      [outputsArray_[i].mpsndarray readBytes:outputs[i]->mutable_data_ptr<uint8_t>()
                                strideBytes:nil];
  }

  ET_CHECK_OR_RETURN_ERROR(
    err == Error::Ok,
    Internal,
    "Could not synchronize on the MPSStream");

  return Error::Ok;
}

} // namespace delegate
} // namespace mps
} // namespace executor
} // namespace torch
