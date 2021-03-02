// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "simple_tensor_allocator.h"
#include "tensorprotoutils.h"

namespace onnxruntime {
common::Status SimpleTensorAllocator::Trace(int id, const ONNX_NAMESPACE::TensorProto* value) {
  values_[id] = value;
  return Status::OK();
}

common::Status SimpleTensorAllocator::GetPreallocatedBuffer(int ort_value_index, const char* name,
                                                            std::unique_ptr<MemBuffer>& out) {
  const struct OrtMemoryInfo& location = seq_plan_.GetLocation(ort_value_index);
  auto iter = values_.find(ort_value_index);
  if (iter == values_.end()) {
    // not traced, return empty buffer with allocator, so that the others can
    // allocate buffer by their own
    out = onnxruntime::make_unique<MemBuffer>(GetAllocator(location));
    return Status::OK();
  }

  size_t len = 0;
  ORT_RETURN_IF_ERROR(utils::GetSizeInBytesFromTensorProto<kAllocAlignment>(*iter->second, &len));
  if (len == 0) {
    out = onnxruntime::make_unique<MemBuffer>(nullptr, 0, location);
    return Status::OK();
  }
  auto alloc = GetAllocator(location);
  if (!alloc)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get allocator for initializer '", name,
                           "', location: ", location.ToString());
  void* buffer = alloc->Alloc(len);
  weights_buffers_.push_back(BufferUniquePtr(buffer, alloc));
  out = onnxruntime::make_unique<MemBuffer>(buffer, len, location);
  return Status::OK();
}
}  // namespace onnxruntime
