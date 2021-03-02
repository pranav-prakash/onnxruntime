// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"

namespace onnxruntime {
/**
 * A simple POD for using with tensor deserialization
 */
class MemBuffer {
 public:
  MemBuffer(void* buffer, size_t len, const OrtMemoryInfo& alloc_info)
      : buffer_(buffer), len_(len), alloc_info_(alloc_info) {}

  MemBuffer(AllocatorPtr&& alloc) : buffer_(0), len_(0), alloc_(std::move(alloc)), alloc_info_(alloc_->Info()) {}

  void* GetBuffer() const { return buffer_; }

  size_t GetLen() const { return len_; }
  const OrtMemoryInfo& GetAllocInfo() const { return alloc_info_; }

  const AllocatorPtr& GetAllocator() const {
    return alloc_;
  }

 private:
  void* const buffer_;
  const size_t len_;
  AllocatorPtr alloc_;
  const OrtMemoryInfo& alloc_info_;
};
};  // namespace onnxruntime
