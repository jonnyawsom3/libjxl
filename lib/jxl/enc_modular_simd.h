// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_MODULAR_SIMD_H_
#define LIB_JXL_ENC_MODULAR_SIMD_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"

namespace jxl {

// Estimates the modular residual cost for the inclusive channel range
// [begin_c, end_c]. By default all channels and the clamped-gradient
// predictor are used. `use_zero_predictor` models streams whose residual
// channels are encoded with Predictor::Zero.
StatusOr<float> EstimateCost(
    const Image& img, size_t begin_c = 0,
    size_t end_c = static_cast<size_t>(-1), bool use_zero_predictor = false);

namespace estimate_cost_detail {
constexpr size_t kLastThreshold = 501;
constexpr size_t kLastCtx = 16;
const std::array<uint8_t, kLastThreshold>& ContextMap();
}  // namespace estimate_cost_detail

}  // namespace jxl

#endif  // LIB_JXL_ENC_MODULAR_SIMD_H_
