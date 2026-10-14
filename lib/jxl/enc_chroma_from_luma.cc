// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_chroma_from_luma.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <limits>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quant_weights.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_chroma_from_luma.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/dec_transforms-inl.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/simd_util.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::SumOfLanes;
using hwy::HWY_NAMESPACE::Zero;

static HWY_FULL(float) df;

// A simple radial contrast-sensitivity profile: retain more influence for
// low-frequency AC, where chroma prediction errors are more visible.
struct WeightProfile {
  float w[64];
  WeightProfile() {
    constexpr float kMaxRadius = 9.8994949366f;  // sqrt(7^2 + 7^2)
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        const float radius = std::sqrt(static_cast<float>(x * x + y * y));
        w[y * 8 + x] = (x == 0 && y == 0)
                           ? 0.0f
                           : 1.0f + 2.0f * (1.0f - radius / kMaxRadius);
      }
    }
  }
};
static const WeightProfile kWeightProfile;

// Chooses a CfL multiplier by minimizing the friend's frequency-weighted L1
// residual and rate proxy. The map predictor must come from an immutable map
// snapshot: ComputeTile runs concurrently, so reading the live map would race.
int32_t FindBestMultiplierWeighted(
    const float* values_m, const float* values_s, const float* weights,
    size_t num, float weight_sum, float base, float tile_q, float dc_avg,
    int pred, const CompressParams& cparams, bool fast, float* scratch_space) {
  if (num == 0 || weight_sum <= 0.0f) {
    return (std::abs(pred) <= 4) ? pred : 0;
  }

  const size_t lanes = Lanes(df);
  const auto base_v = Set(df, base);
  auto dot_mr_v = Zero(df);
  auto dot_mm_v = Zero(df);
  auto dot_rr_v = Zero(df);
  auto abs_r_unweighted_v = Zero(df);

  // Fit the adjustment to the existing base correlation. For B, this is
  // essential: the regression must model (S - base * M), not S itself.
  for (size_t i = 0; i < num; i += lanes) {
    const auto m = LoadU(df, values_m + i);
    const auto s = LoadU(df, values_s + i);
    const auto w = LoadU(df, weights + i);
    const auto residual0 = Sub(s, Mul(base_v, m));
    dot_mr_v = MulAdd(w, Mul(m, residual0), dot_mr_v);
    dot_mm_v = MulAdd(w, Mul(m, m), dot_mm_v);
    dot_rr_v = MulAdd(w, Mul(residual0, residual0), dot_rr_v);
    abs_r_unweighted_v = Add(abs_r_unweighted_v, Abs(residual0));
  }
  const float dot_mr = GetLane(SumOfLanes(df, dot_mr_v));
  const float dot_mm = GetLane(SumOfLanes(df, dot_mm_v));
  const float dot_rr = GetLane(SumOfLanes(df, dot_rr_v));
  const float sum_abs_r_unweighted =
      GetLane(SumOfLanes(df, abs_r_unweighted_v));
  const float energy = sum_abs_r_unweighted / static_cast<float>(num);
  const float correlation =
      (dot_mm > 1e-8f && dot_rr > 1e-8f)
          ? std::abs(dot_mr) / std::sqrt(dot_mm * dot_rr)
          : 0.0f;
  const float target_delta = dot_mm > 1e-8f ? dot_mr / dot_mm : 0.0f;
  const int target = jxl::Clamp1(
      static_cast<int>(std::round(target_delta * kDefaultColorFactor)),
      -128, 127);

  // Retain the friend's low-energy suppression, but calculate both the
  // correlation and energy relative to the actual no-CfL residual above.
  float deadzone_penalty = 1.0f;
  const float dc_neutral_threshold = (base == 0.0f) ? 0.10f : 0.35f;
  if (correlation < 0.20f && energy < 0.10f &&
      dc_avg < dc_neutral_threshold) {
    deadzone_penalty = 2.5f;
  } else if (correlation < 0.35f && energy < 0.05f &&
             dc_avg < 1.5f * dc_neutral_threshold) {
    deadzone_penalty = 1.8f;
  }

  // The friend's native objective is weighted L1 residual in its own
  // coefficient scale (q = Scale() * raw_quant_field). Do not divide this by
  // weight_sum or reapply the restored 0.8.x 400x scale: either operation
  // changes the relative strength of the friend's rate terms.
  constexpr float kOversatPenalty = 1.2f;
  auto evaluate = [&](const float* candidates, float* costs,
                      size_t candidate_count) {
    const auto oversat_v = Set(df, kOversatPenalty);
    const auto inv_color_factor = Set(df, 1.0f / kDefaultColorFactor);
    for (size_t c = 0; c < candidate_count; c += lanes) {
      const auto candidate_v = LoadU(df, candidates + c);
      const auto factor_v = Add(base_v, Mul(candidate_v, inv_color_factor));
      auto sum_v = Zero(df);
      for (size_t i = 0; i < num; ++i) {
        const auto m = Set(df, values_m[i]);
        const auto chroma = Set(df, values_s[i]);
        const auto w = Set(df, weights[i]);
        const auto baseline = Sub(chroma, Mul(base_v, m));
        const auto residual = Sub(chroma, Mul(factor_v, m));
        auto cost = Mul(w, Abs(residual));
        const auto worsens_residual = Lt(Abs(baseline), Abs(residual));
        cost = IfThenElse(worsens_residual, Mul(cost, oversat_v), cost);
        sum_v = Add(sum_v, cost);
      }
      StoreU(sum_v, df, costs + c);
    }
  };

  // Approximate map coding cost from candidate - predictor. The predictor is
  // read from an immutable snapshot prepared between parallel encoder phases.
  auto estimate_bits = [](int candidate, int predictor) {
    const int residual = std::abs(candidate - predictor);
    if (residual == 0) return 0.8f;
    if (residual == 1) return 2.0f;
    return 2.5f + std::log2(static_cast<float>(residual));
  };
  const float lambda_bits = 0.005f * tile_q;
  const float multiplier_bit_cost = 0.05f * tile_q;
  auto score = [&](int candidate, float distortion_sum) {
    return distortion_sum + lambda_bits * estimate_bits(candidate, pred) +
           std::abs(candidate) * multiplier_bit_cost * deadzone_penalty;
  };

  int coarse_step = 1;
  if (fast || cparams.speed_tier >= SpeedTier::kCheetah) {
    coarse_step = 8;
  } else if (cparams.speed_tier >= SpeedTier::kHare) {
    coarse_step = 4;
  } else if (cparams.speed_tier >= SpeedTier::kSquirrel) {
    coarse_step = 2;
  }

  // Reuse the transform scratch after coefficient extraction. Each SIMD lane
  // represents a candidate multiplier, not an encoder thread or neighbouring
  // tile, so the optimizer remains safe under RunOnPool.
  constexpr size_t kCandidateStride = 384;
  float* candidates = scratch_space;
  float* costs = candidates + kCandidateStride;
  float* fine_candidates = costs + kCandidateStride;
  float* fine_costs = fine_candidates + kCandidateStride;

  size_t num_candidates = 0;
  auto add_candidate = [&](int candidate) {
    candidate = jxl::Clamp1(candidate, -128, 127);
    for (size_t i = 0; i < num_candidates; ++i) {
      if (static_cast<int>(candidates[i]) == candidate) return;
    }
    candidates[num_candidates++] = static_cast<float>(candidate);
  };
  for (int candidate = -128; candidate <= 127; candidate += coarse_step) {
    add_candidate(candidate);
  }
  add_candidate(127);
  add_candidate(0);
  add_candidate(pred);
  add_candidate(target);
  for (int d = -4; d <= 4; ++d) {
    add_candidate(target + d);
    add_candidate(pred + d);
  }

  size_t padded_candidates = num_candidates;
  while (padded_candidates % lanes != 0) {
    candidates[padded_candidates++] = candidates[0];
  }
  evaluate(candidates, costs, padded_candidates);

  int best_candidate = 0;
  float best_cost = std::numeric_limits<float>::max();
  for (size_t i = 0; i < num_candidates; ++i) {
    const int candidate = static_cast<int>(candidates[i]);
    const float cost = score(candidate, costs[i]);
    if (cost < best_cost) {
      best_cost = cost;
      best_candidate = candidate;
    }
  }

  // Refine around the best candidate and least-squares target. Match the
  // friend's search radius (one coarse step in either direction).
  if (coarse_step > 1) {
    size_t num_fine = 0;
    auto add_fine_candidate = [&](int candidate) {
      if (candidate < -128 || candidate > 127) return;
      for (size_t i = 0; i < num_fine; ++i) {
        if (static_cast<int>(fine_candidates[i]) == candidate) return;
      }
      fine_candidates[num_fine++] = static_cast<float>(candidate);
    };
    for (int d = -coarse_step; d <= coarse_step; ++d) {
      add_fine_candidate(best_candidate + d);
      add_fine_candidate(target + d);
    }
    size_t padded_fine = num_fine;
    while (padded_fine % lanes != 0) {
      fine_candidates[padded_fine++] = fine_candidates[0];
    }
    if (num_fine != 0) {
      evaluate(fine_candidates, fine_costs, padded_fine);
      for (size_t i = 0; i < num_fine; ++i) {
        const int candidate = static_cast<int>(fine_candidates[i]);
        const float cost = score(candidate, fine_costs[i]);
        if (cost < best_cost) {
          best_cost = cost;
          best_candidate = candidate;
        }
      }
    }
  }
  return best_candidate;
}
Status InitDCStorage(JxlMemoryManager* memory_manager, size_t num_blocks,
                     ImageF* dc_values) {
  // First row: Y channel
  // Second row: X channel
  // Third row: Y channel
  // Fourth row: B channel
  JXL_ASSIGN_OR_RETURN(
      *dc_values,
      ImageF::Create(memory_manager, RoundUpTo(num_blocks, Lanes(df)), 4));

  JXL_ENSURE(dc_values->xsize() != 0);
  // Zero-fill the last lanes
  for (size_t y = 0; y < 4; y++) {
    for (size_t x = dc_values->xsize() - Lanes(df); x < dc_values->xsize();
         x++) {
      dc_values->Row(y)[x] = 0;
    }
  }
  return true;
}

Status ComputeTile(const Image3F& opsin, const Rect& opsin_rect,
                   const DequantMatrices& dequant,
                   const AcStrategyImage* ac_strategy,
                   const ImageI* raw_quant_field, const Quantizer* quantizer,
                   const CompressParams& cparams, const Rect& rect, bool fast,
                   bool use_dct8, ImageSB* map_x, ImageSB* map_b,
                   const ImageSB* predictor_map_x,
                   const ImageSB* predictor_map_b, ImageF* dc_values,
                   Span<float> mem) {
  static_assert(kEncTileDimInBlocks == kColorTileDimInBlocks,
                "Invalid color tile dim");
  size_t xsize_blocks = opsin_rect.xsize() / kBlockDim;
  const size_t dct_scratch_size =
      3 * (MaxVectorSize() / sizeof(float)) * AcStrategy::kMaxBlockDim;

  const size_t y0 = rect.y0();
  const size_t x0 = rect.x0();
  const size_t x1 = rect.x0() + rect.xsize();
  const size_t y1 = rect.y0() + rect.ysize();

  int ty = y0 / kColorTileDimInBlocks;
  int tx = x0 / kColorTileDimInBlocks;

  int8_t* JXL_RESTRICT row_out_x = map_x->Row(ty);
  int8_t* JXL_RESTRICT row_out_b = map_b->Row(ty);

  float* JXL_RESTRICT dc_values_yx = dc_values->Row(0);
  float* JXL_RESTRICT dc_values_x = dc_values->Row(1);
  float* JXL_RESTRICT dc_values_yb = dc_values->Row(2);
  float* JXL_RESTRICT dc_values_b = dc_values->Row(3);

  // All are aligned.
  float* HWY_RESTRICT block_y = mem.begin();
  float* HWY_RESTRICT block_x = block_y + AcStrategy::kMaxCoeffArea;
  float* HWY_RESTRICT block_b = block_x + AcStrategy::kMaxCoeffArea;
  JXL_ENSURE(mem.remove_prefix(3 * AcStrategy::kMaxCoeffArea));
  float* HWY_RESTRICT coeffs_yx = mem.begin();
  float* HWY_RESTRICT coeffs_x = coeffs_yx + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_yb = coeffs_x + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_b = coeffs_yb + kColorTileDim * kColorTileDim;
  float* HWY_RESTRICT coeffs_w = coeffs_b + kColorTileDim * kColorTileDim;
  JXL_ENSURE(mem.remove_prefix(5 * kColorTileDim * kColorTileDim));
  constexpr size_t dc_size =
      AcStrategy::kMaxCoeffBlocks * AcStrategy::kMaxCoeffBlocks;
  float* HWY_RESTRICT dc_y = mem.begin();
  float* HWY_RESTRICT dc_x = dc_y + dc_size;
  float* HWY_RESTRICT dc_b = dc_x + dc_size;
  JXL_ENSURE(mem.remove_prefix(3 * dc_size));
  float* HWY_RESTRICT scratch_space = mem.begin();
  JXL_ENSURE(mem.size() == 2 * AcStrategy::kMaxCoeffArea + dct_scratch_size);

  size_t num_ac = 0;
  float tile_q_sum = 0.0f;
  size_t tile_q_blocks = 0;
  float coeff_weight_sum = 0.0f;

  for (size_t y = y0; y < y1; ++y) {
    const float* JXL_RESTRICT row_y =
        opsin_rect.ConstPlaneRow(opsin, 1, y * kBlockDim);
    const float* JXL_RESTRICT row_x =
        opsin_rect.ConstPlaneRow(opsin, 0, y * kBlockDim);
    const float* JXL_RESTRICT row_b =
        opsin_rect.ConstPlaneRow(opsin, 2, y * kBlockDim);
    size_t stride = opsin.PixelsPerRow();

    for (size_t x = x0; x < x1; x++) {
      AcStrategy acs = use_dct8
                           ? AcStrategy::FromRawStrategy(AcStrategyType::DCT)
                           : ac_strategy->ConstRow(y)[x];
      if (!acs.IsFirstBlock()) continue;
      size_t xs = acs.covered_blocks_x();
      TransformFromPixels(acs.Strategy(), row_y + x * kBlockDim, stride,
                          block_y, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_y, dc_y, xs, scratch_space);
      TransformFromPixels(acs.Strategy(), row_x + x * kBlockDim, stride,
                          block_x, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_x, dc_x, xs, scratch_space);
      TransformFromPixels(acs.Strategy(), row_b + x * kBlockDim, stride,
                          block_b, scratch_space);
      DCFromLowestFrequencies(acs.Strategy(), block_b, dc_b, xs, scratch_space);
      const float* const JXL_RESTRICT qm_x =
          dequant.InvMatrix(acs.Strategy(), 0);
      const float* const JXL_RESTRICT qm_b =
          dequant.InvMatrix(acs.Strategy(), 2);
      float q_dc_x = use_dct8 ? 1 : 1.0f / quantizer->GetInvDcStep(0);
      float q_dc_b = use_dct8 ? 1 : 1.0f / quantizer->GetInvDcStep(2);

      // Copy DCs in dc_values.
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < xs; ix++) {
          dc_values_yx[(iy + y) * xsize_blocks + ix + x] =
              dc_y[iy * xs + ix] * q_dc_x;
          dc_values_x[(iy + y) * xsize_blocks + ix + x] =
              dc_x[iy * xs + ix] * q_dc_x;
          dc_values_yb[(iy + y) * xsize_blocks + ix + x] =
              dc_y[iy * xs + ix] * q_dc_b;
          dc_values_b[(iy + y) * xsize_blocks + ix + x] =
              dc_b[iy * xs + ix] * q_dc_b;
        }
      }

      // Do not use this block for computing AC CfL.
      if (acs.covered_blocks_x() + x0 > x1 ||
          acs.covered_blocks_y() + y0 > y1) {
        continue;
      }

      // Copy AC coefficients in the local block. The order in which
      // coefficients get stored does not matter.
      size_t cx = acs.covered_blocks_x();
      size_t cy = acs.covered_blocks_y();
      CoefficientLayout(&cy, &cx);
      // Zero out LFs. This introduces terms in the optimization loop that
      // don't affect the result, as they are all 0, but allow for simpler
      // SIMDfication.
      for (size_t iy = 0; iy < cy; iy++) {
        for (size_t ix = 0; ix < cx; ix++) {
          block_y[cx * kBlockDim * iy + ix] = 0;
          block_x[cx * kBlockDim * iy + ix] = 0;
          block_b[cx * kBlockDim * iy + ix] = 0;
        }
      }
      const float qq =
          (raw_quant_field == nullptr) ? 1.0f : raw_quant_field->Row(y)[x];
      // This candidate uses the friend's native objective and scale. The
      // old 0.8.x *400 scale belongs to the quadratic-like loss and must not
      // be mixed into this weighted-L1 objective.
      const float q = use_dct8 ? 1.0f : quantizer->Scale() * qq;
      const size_t block_count =
          acs.covered_blocks_x() * acs.covered_blocks_y();
      float q_sum = 0.0f;
      for (size_t qy = 0; qy < acs.covered_blocks_y(); ++qy) {
        for (size_t qx = 0; qx < acs.covered_blocks_x(); ++qx) {
          q_sum += (raw_quant_field == nullptr)
                       ? 1.0f
                       : raw_quant_field->Row(y + qy)[x + qx];
        }
      }
      const float area_q = use_dct8
                               ? 1.0f
                               : quantizer->Scale() * q_sum /
                                     static_cast<float>(block_count);
      tile_q_sum += area_q * block_count;
      tile_q_blocks += block_count;
      const auto qv = Set(df, q);

      // CfL coefficients are packed using the same canonical layout as the
      // transform output. For larger transforms, collapse their frequency
      // grid back to the 8x8 sensitivity profile.
      const size_t sx = cx * kBlockDim;
      const size_t sy = cy * kBlockDim;
      const size_t shift_x = static_cast<size_t>(
          __builtin_ctz(static_cast<unsigned int>(cx)));
      const size_t shift_y = static_cast<size_t>(
          __builtin_ctz(static_cast<unsigned int>(cy)));
      for (size_t iy = 0; iy < sy; ++iy) {
        const size_t iy_norm = iy >> shift_y;
        for (size_t ix = 0; ix < sx; ++ix) {
          const size_t ix_norm = ix >> shift_x;
          const float w = kWeightProfile.w[iy_norm * 8 + ix_norm];
          coeffs_w[num_ac + iy * sx + ix] = w;
          coeff_weight_sum += w;
        }
      }

      for (size_t i = 0; i < cx * cy * 64; i += Lanes(df)) {
        const auto b_y = Load(df, block_y + i);
        const auto b_x = Load(df, block_x + i);
        const auto b_b = Load(df, block_b + i);
        const auto qqm_x = Mul(qv, Load(df, qm_x + i));
        const auto qqm_b = Mul(qv, Load(df, qm_b + i));
        Store(Mul(b_y, qqm_x), df, coeffs_yx + num_ac);
        Store(Mul(b_x, qqm_x), df, coeffs_x + num_ac);
        Store(Mul(b_y, qqm_b), df, coeffs_yb + num_ac);
        Store(Mul(b_b, qqm_b), df, coeffs_b + num_ac);
        num_ac += Lanes(df);
      }
    }
  }
  JXL_ENSURE(num_ac % Lanes(df) == 0);
  const float tile_q = tile_q_blocks == 0
                           ? 1.0f
                           : tile_q_sum / static_cast<float>(tile_q_blocks);
  float dc_sum_x = 0.0f;
  float dc_sum_b = 0.0f;
  size_t dc_count = 0;
  for (size_t y = y0; y < y1; ++y) {
    for (size_t x = x0; x < x1; ++x) {
      const size_t idx = y * xsize_blocks + x;
      dc_sum_x += std::abs(dc_values_x[idx]);
      dc_sum_b += std::abs(dc_values_b[idx] -
                           jxl::cms::kYToBRatio * dc_values_yb[idx]);
      ++dc_count;
    }
  }
  const float dc_avg_x =
      dc_count == 0 ? 0.0f : dc_sum_x / static_cast<float>(dc_count);
  const float dc_avg_b =
      dc_count == 0 ? 0.0f : dc_sum_b / static_cast<float>(dc_count);
  auto get_pred = [](const ImageSB* map, int tx, int ty) {
    if (map == nullptr || (tx == 0 && ty == 0)) return 0;
    if (tx == 0) return static_cast<int>(map->Row(ty - 1)[tx]);
    if (ty == 0) return static_cast<int>(map->Row(ty)[tx - 1]);
    const int left = map->Row(ty)[tx - 1];
    const int top = map->Row(ty - 1)[tx];
    const int top_left = map->Row(ty - 1)[tx - 1];
    const int gradient = left + top - top_left;
    return jxl::Clamp1(gradient, std::min(left, top), std::max(left, top));
  };
  const int pred_x = get_pred(predictor_map_x, tx, ty);
  const int pred_b = get_pred(predictor_map_b, tx, ty);
  row_out_x[tx] = FindBestMultiplierWeighted(
      coeffs_yx, coeffs_x, coeffs_w, num_ac, coeff_weight_sum, 0.0f, tile_q,
      dc_avg_x, pred_x, cparams, fast, scratch_space);
  row_out_b[tx] = FindBestMultiplierWeighted(
      coeffs_yb, coeffs_b, coeffs_w, num_ac, coeff_weight_sum,
      jxl::cms::kYToBRatio, tile_q, dc_avg_b, pred_b, cparams, fast,
      scratch_space);
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(InitDCStorage);
HWY_EXPORT(ComputeTile);

Status CfLHeuristics::Init(const Rect& rect) {
  size_t xsize_blocks = rect.xsize() / kBlockDim;
  size_t ysize_blocks = rect.ysize() / kBlockDim;
  return HWY_DYNAMIC_DISPATCH(InitDCStorage)(
      memory_manager, xsize_blocks * ysize_blocks, &dc_values);
}

Status CfLHeuristics::ComputeTile(const Rect& r, const Image3F& opsin,
                                  const Rect& opsin_rect,
                                  const DequantMatrices& dequant,
                                  const AcStrategyImage* ac_strategy,
                                  const ImageI* raw_quant_field,
                                  const Quantizer* quantizer,
                                  const CompressParams& cparams, bool fast,
                                  size_t thread, ColorCorrelationMap* cmap,
                                  const ImageSB* predictor_map_x,
                                  const ImageSB* predictor_map_b) {
  bool use_dct8 = ac_strategy == nullptr;
  Span<float> scratch(mem.address<float>() + thread * ItemsPerThread(),
                      ItemsPerThread());
  return HWY_DYNAMIC_DISPATCH(ComputeTile)(
      opsin, opsin_rect, dequant, ac_strategy, raw_quant_field, quantizer,
      cparams, r, fast, use_dct8, &cmap->ytox_map, &cmap->ytob_map,
      predictor_map_x, predictor_map_b, &dc_values, scratch);
}

Status ColorCorrelationEncodeDC(const ColorCorrelation& color_correlation,
                                BitWriter* writer, LayerType layer,
                                AuxOut* aux_out) {
  float color_factor = color_correlation.GetColorFactor();
  float base_correlation_x = color_correlation.GetBaseCorrelationX();
  float base_correlation_b = color_correlation.GetBaseCorrelationB();
  int32_t ytox_dc = color_correlation.GetYToXDC();
  int32_t ytob_dc = color_correlation.GetYToBDC();

  return writer->WithMaxBits(
      1 + 2 * kBitsPerByte + 12 + 32, layer, aux_out, [&]() -> Status {
        if (ytox_dc == 0 && ytob_dc == 0 &&
            color_factor == kDefaultColorFactor && base_correlation_x == 0.0f &&
            base_correlation_b == jxl::cms::kYToBRatio) {
          writer->Write(1, 1);
          return true;
        }
        writer->Write(1, 0);
        JXL_RETURN_IF_ERROR(
            U32Coder::Write(kColorFactorDist, color_factor, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_x, writer));
        JXL_RETURN_IF_ERROR(F16Coder::Write(base_correlation_b, writer));
        writer->Write(kBitsPerByte,
                      ytox_dc - std::numeric_limits<int8_t>::min());
        writer->Write(kBitsPerByte,
                      ytob_dc - std::numeric_limits<int8_t>::min());
        return true;
      });
}

}  // namespace jxl
#endif  // HWY_ONCE
