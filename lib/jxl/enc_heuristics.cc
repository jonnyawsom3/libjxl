// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_heuristics.h"

#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_ac_strategy.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_chroma_from_luma.h"
#include "lib/jxl/enc_gaborish.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_noise.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/enc_splines.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"

namespace jxl {

struct AuxOut;

void FindBestBlockEntropyModel(const CompressParams& cparams, const ImageI& rqf,
                               const AcStrategyImage& ac_strategy,
                               BlockCtxMap* block_ctx_map) {
  if (cparams.decoding_speed_tier >= 1) {
    static constexpr uint8_t kSimpleCtxMap[] = {
        // Cluster all blocks together
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    static_assert(
        3 * kNumOrders == sizeof(kSimpleCtxMap) / sizeof *kSimpleCtxMap,
        "Update simple context map");

    auto& bcm = *block_ctx_map;
    bcm.ctx_map.assign(std::begin(kSimpleCtxMap), std::end(kSimpleCtxMap));
    bcm.num_ctxs = 2;
    bcm.num_dc_ctxs = 1;
    return;
  }
  if (cparams.speed_tier >= SpeedTier::kFalcon) {
    return;
  }
  size_t tot = rqf.xsize() * rqf.ysize();
  size_t size_for_ctx_model = (1 << 10) * cparams.butteraugli_distance;
  if (tot < size_for_ctx_model) return;

  struct OccCounters {
    OccCounters(const ImageI& rqf, const AcStrategyImage& ac_strategy) {
      for (size_t y = 0; y < rqf.ysize(); y++) {
        const int32_t* qf_row = rqf.Row(y);
        AcStrategyRow acs_row = ac_strategy.ConstRow(y);
        for (size_t x = 0; x < rqf.xsize(); x++) {
          int ord = kStrategyOrder[acs_row[x].RawStrategy()];
          int qf = qf_row[x] - 1;
          qf_counts[qf]++;
          qf_ord_counts[ord][qf]++;
          ord_counts[ord]++;
        }
      }
    }

    size_t qf_counts[256] = {};
    size_t qf_ord_counts[kNumOrders][256] = {};
    size_t ord_counts[kNumOrders] = {};
  };
  std::unique_ptr<OccCounters> counters(new OccCounters(rqf, ac_strategy));

  size_t size_for_qf_split = (1 << 13) * cparams.butteraugli_distance;
  size_t num_qf_segments = tot < size_for_qf_split ? 1 : 2;
  std::vector<uint32_t>& qft = block_ctx_map->qf_thresholds;
  qft.clear();
  size_t cumsum = 0;
  size_t next = 1;
  size_t last_cut = 256;
  size_t cut = tot * next / num_qf_segments;
  for (uint32_t j = 0; j < 256; j++) {
    cumsum += counters->qf_counts[j];
    if (cumsum > cut) {
      if (j != 0) qft.push_back(j);
      last_cut = j;
      while (cumsum > cut) {
        next++;
        cut = tot * next / num_qf_segments;
      }
    } else if (next > qft.size() + 1) {
      if (j - 1 == last_cut && j != 0) qft.push_back(j);
    }
  }

  std::vector<size_t> counts(kNumOrders * (qft.size() + 1));
  size_t qft_pos = 0;
  for (size_t j = 0; j < 256; j++) {
    if (qft_pos < qft.size() && j == qft[qft_pos]) qft_pos++;
    for (size_t i = 0; i < kNumOrders; i++) {
      counts[qft_pos + i * (qft.size() + 1)] += counters->qf_ord_counts[i][j];
    }
  }

  std::vector<uint8_t> remap((qft.size() + 1) * kNumOrders);
  std::iota(remap.begin(), remap.end(), 0);
  std::vector<uint8_t> clusters(remap);
  size_t nb_clusters =
      Clamp1(static_cast<int>(tot / size_for_ctx_model / 2), 2, 9);
  size_t nb_clusters_chroma =
      Clamp1(static_cast<int>(tot / size_for_ctx_model / 3), 1, 5);
  while (clusters.size() > nb_clusters) {
    std::sort(clusters.begin(), clusters.end(),
              [&](int a, int b) { return counts[a] > counts[b]; });
    counts[clusters[clusters.size() - 2]] += counts[clusters.back()];
    counts[clusters.back()] = 0;
    remap[clusters.back()] = clusters[clusters.size() - 2];
    clusters.pop_back();
  }
  for (size_t i = 0; i < remap.size(); i++) {
    while (remap[remap[i]] != remap[i]) remap[i] = remap[remap[i]];
  }
  std::vector<uint8_t> remap_remap(remap.size(), remap.size());
  size_t num = 0;
  for (size_t i = 0; i < remap.size(); i++) {
    if (remap_remap[remap[i]] == remap.size()) remap_remap[remap[i]] = num++;
    remap[i] = remap_remap[remap[i]];
  }
  auto& ctx_map = block_ctx_map->ctx_map;
  ctx_map = remap;
  ctx_map.resize(remap.size() * 3);
  for (size_t i = remap.size(); i < remap.size() * 3; i++) {
    ctx_map[i] = num + Clamp1(static_cast<int>(remap[i % remap.size()]), 0,
                              static_cast<int>(nb_clusters_chroma) - 1);
  }
  block_ctx_map->num_ctxs =
      *std::max_element(ctx_map.begin(), ctx_map.end()) + 1;
}

namespace {

Status FindBestDequantMatrices(JxlMemoryManager* memory_manager,
                               const CompressParams& cparams,
                               ModularFrameEncoder* modular_frame_encoder,
                               DequantMatrices* dequant_matrices) {
  *dequant_matrices = DequantMatrices();
  if (cparams.max_error_mode || cparams.disable_perceptual_optimizations) {
    constexpr float kMSEWeights[3] = {0.001, 0.001, 0.001};
    const float* wp = cparams.disable_perceptual_optimizations
                          ? kMSEWeights
                          : cparams.max_error;
    float weights[3][1] = {{1.0f / wp[0]}, {1.0f / wp[1]}, {1.0f / wp[2]}};
    DctQuantWeightParams dct_params(weights);
    std::vector<QuantEncoding> encodings(kNumQuantTables,
                                         QuantEncoding::DCT(dct_params));
    JXL_RETURN_IF_ERROR(DequantMatricesSetCustom(dequant_matrices, encodings,
                                                 modular_frame_encoder));
    float dc_weights[3] = {1.0f / wp[0], 1.0f / wp[1], 1.0f / wp[2]};
    JXL_RETURN_IF_ERROR(DequantMatricesSetCustomDC(
        memory_manager, dequant_matrices, dc_weights));
  }
  return true;
}

void StoreMin2(const float v, float& min1, float& min2) {
  if (v < min2) {
    if (v < min1) {
      min2 = min1;
      min1 = v;
    } else {
      min2 = v;
    }
  }
}

void CreateMask(const ImageF& image, ImageF& mask) {
  for (size_t y = 0; y < image.ysize(); y++) {
    const auto* row_n = y > 0 ? image.Row(y - 1) : image.Row(y);
    const auto* row_in = image.Row(y);
    const auto* row_s = y + 1 < image.ysize() ? image.Row(y + 1) : image.Row(y);
    auto* row_out = mask.Row(y);
    for (size_t x = 0; x < image.xsize(); x++) {
      float c = row_in[x];
      float w = x > 0 ? row_in[x - 1] : row_in[x];
      float e = x + 1 < image.xsize() ? row_in[x + 1] : row_in[x];
      float n = row_n[x];
      float s = row_s[x];
      float dw = std::abs(c - w);
      float de = std::abs(c - e);
      float dn = std::abs(c - n);
      float ds = std::abs(c - s);
      float min = std::numeric_limits<float>::max();
      float min2 = std::numeric_limits<float>::max();
      StoreMin2(dw, min, min2);
      StoreMin2(de, min, min2);
      StoreMin2(dn, min, min2);
      StoreMin2(ds, min, min2);
      row_out[x] = min2;
    }
  }
}

Status DownsampleImage2_Sharper(const ImageF& input, ImageF* output) {
  const int64_t kernelx = 12;
  const int64_t kernely = 12;
  JxlMemoryManager* memory_manager = input.memory_manager();

  static const float kernel[144] = {
      -0.000314256996835f, -0.000314256996835f, -0.000897597057705f,
      -0.000562751488849f, -0.000176807273646f, 0.001864627368902f,
      0.001864627368902f, -0.000176807273646f, -0.000562751488849f,
      -0.000897597057705f, -0.000314256996835f, -0.000314256996835f,
      -0.000314256996835f, -0.001527942804748f, -0.000121760530512f,
      0.000191123989093f, 0.010193185932466f, 0.058637519197110f,
      0.058637519197110f, 0.010193185932466f, 0.000191123989093f,
      -0.000121760530512f, -0.001527942804748f, -0.000314256996835f,
      -0.000897597057705f, -0.000121760530512f, 0.000946363683751f,
      0.007113577630288f, 0.000437956841058f, -0.000372823835211f,
      -0.000372823835211f, 0.000437956841058f, 0.007113577630288f,
      0.000946363683751f, -0.000121760530512f, -0.000897597057705f,
      -0.000562751488849f, 0.000191123989093f, 0.007113577630288f,
      0.044592622228814f, 0.000222278879007f, -0.162864473015945f,
      -0.162864473015945f, 0.000222278879007f, 0.044592622228814f,
      0.007113577630288f, 0.000191123989093f, -0.000562751488849f,
      -0.000176807273646f, 0.010193185932466f, 0.000437956841058f,
      0.000222278879007f, -0.000913092543974f, -0.017071696107902f,
      -0.017071696107902f, -0.000913092543974f, 0.000222278879007f,
      0.000437956841058f, 0.010193185932466f, -0.000176807273646f,
      0.001864627368902f, 0.058637519197110f, -0.000372823835211f,
      -0.162864473015945f, -0.017071696107902f, 0.414660099370354f,
      0.414660099370354f, -0.017071696107902f, -0.162864473015945f,
      -0.000372823835211f, 0.058637519197110f, 0.001864627368902f,
      0.001864627368902f, 0.058637519197110f, -0.000372823835211f,
      -0.162864473015945f, -0.017071696107902f, 0.414660099370354f,
      0.414660099370354f, -0.017071696107902f, -0.162864473015945f,
      -0.000372823835211f, 0.058637519197110f, 0.001864627368902f,
      -0.000176807273646f, 0.010193185932466f, 0.000437956841058f,
      0.000222278879007f, -0.000913092543974f, -0.017071696107902f,
      -0.017071696107902f, -0.000913092543974f, 0.000222278879007f,
      0.000437956841058f, 0.010193185932466f, -0.000176807273646f,
      -0.000562751488849f, 0.000191123989093f, 0.007113577630288f,
      0.044592622228814f, 0.000222278879007f, -0.162864473015945f,
      -0.162864473015945f, 0.000222278879007f, 0.044592622228814f,
      0.007113577630288f, 0.000191123989093f, -0.000562751488849f,
      -0.000897597057705f, -0.000121760530512f, 0.000946363683751f,
      0.007113577630288f, 0.000437956841058f, -0.000372823835211f,
      -0.000372823835211f, 0.000437956841058f, 0.007113577630288f,
      0.000946363683751f, -0.000121760530512f, -0.000897597057705f,
      -0.000314256996835f, -0.001527942804748f, -0.000121760530512f,
      0.000191123989093f, 0.010193185932466f, 0.058637519197110f,
      0.058637519197110f, 0.010193185932466f, 0.000191123989093f,
      -0.000121760530512f, -0.001527942804748f, -0.000314256996835f,
      -0.000314256996835f, -0.000314256996835f, -0.000897597057705f,
      -0.000562751488849f, -0.000176807273646f, 0.001864627368902f,
      0.001864627368902f, -0.000176807273646f, -0.000562751488849f,
      -0.000897597057705f, -0.000314256996835f, -0.000314256996835f,
  };

  const int64_t xsize = input.xsize();
  const int64_t ysize = input.ysize();
  JXL_ASSIGN_OR_RETURN(ImageF box_downsample,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(input, &box_downsample));
  JXL_ASSIGN_OR_RETURN(box_downsample, DownsampleImage(box_downsample, 2));

  JXL_ASSIGN_OR_RETURN(ImageF mask,
                       ImageF::Create(memory_manager, box_downsample.xsize(),
                                      box_downsample.ysize()));
  CreateMask(box_downsample, mask);

  for (size_t y = 0; y < output->ysize(); y++) {
    float* row_out = output->Row(y);
    const float* row_in[kernely];
    const float* row_mask = mask.Row(y);
    for (size_t ky = 0; ky < kernely; ky++) {
      int64_t iy = y * 2 + ky - (kernely - 1) / 2;
      if (iy < 0) iy = 0;
      if (iy >= ysize) iy = ysize - 1;
      row_in[ky] = input.Row(iy);
    }

    for (size_t x = 0; x < output->xsize(); x++) {
      float min_value = std::numeric_limits<float>::max();
      float max_value = std::numeric_limits<float>::lowest();
      static constexpr int64_t R = 5;
      for (int64_t ky = R; ky + R < kernely; ky++) {
        for (int64_t kx = R; kx + R < kernelx; kx++) {
          int64_t ix = x * 2 + kx - (kernelx - 1) / 2;
          if (ix < 0) ix = 0;
          if (ix >= xsize) ix = xsize - 1;
          min_value = std::min(min_value, row_in[ky][ix]);
          max_value = std::max(max_value, row_in[ky][ix]);
        }
      }

      float sum = 0;
      for (int64_t ky = 0; ky < kernely; ky++) {
        for (int64_t kx = 0; kx < kernelx; kx++) {
          int64_t ix = x * 2 + kx - (kernelx - 1) / 2;
          if (ix < 0) ix = 0;
          if (ix >= xsize) ix = xsize - 1;
          sum += row_in[ky][ix] * kernel[ky * kernelx + kx];
        }
      }

      row_out[x] = sum;
      const float a = row_mask[x];
      const float clip_min = min_value - a;
      const float clip_max = max_value + a;
      row_out[x] = std::min(clip_max, std::max(clip_min, row_out[x]));
    }
  }
  return true;
}

}  // namespace

Status DownsampleImage2_Sharper(Image3F* opsin) {
  JxlMemoryManager* memory_manager = opsin->memory_manager();
  JXL_ASSIGN_OR_RETURN(
      Image3F downsampled,
      Image3F::Create(memory_manager, DivCeil(opsin->xsize(), 2) + kBlockDim,
                      DivCeil(opsin->ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));

  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DownsampleImage2_Sharper(opsin->Plane(c), &downsampled.Plane(c)));
  }
  *opsin = std::move(downsampled);
  return true;
}

namespace {

const constexpr int64_t kSize = 5;

const float kernel00[25] = {
    -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f, -0.00624645f,
    -0.03452303f, 0.14111091f, 0.28896755f, 0.00278718f, -0.01610267f,
    -0.04022174f, 0.28896755f, 0.56661550f, 0.03777607f, -0.01986694f,
    -0.02921014f, 0.00278718f, 0.03777607f, -0.03144731f, -0.01185068f,
    -0.00624645f, -0.01610267f, -0.01986694f, -0.01185068f, -0.00213539f,
};
const float kernel01[25] = {
    -0.00624645f, -0.01610267f, -0.01986694f, -0.01185068f, -0.00213539f,
    -0.02921014f, 0.00278718f, 0.03777607f, -0.03144731f, -0.01185068f,
    -0.04022174f, 0.28896755f, 0.56661550f, 0.03777607f, -0.01986694f,
    -0.03452303f, 0.14111091f, 0.28896755f, 0.00278718f, -0.01610267f,
    -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f, -0.00624645f,
};
const float kernel10[25] = {
    -0.00624645f, -0.02921014f, -0.04022174f, -0.03452303f, -0.01716200f,
    -0.01610267f, 0.00278718f, 0.28896755f, 0.14111091f, -0.03452303f,
    -0.01986694f, 0.03777607f, 0.56661550f, 0.28896755f, -0.04022174f,
    -0.01185068f, -0.03144731f, 0.03777607f, 0.00278718f, -0.02921014f,
    -0.00213539f, -0.01185068f, -0.01986694f, -0.01610267f, -0.00624645f,
};
const float kernel11[25] = {
    -0.00213539f, -0.01185068f, -0.01986694f, -0.01610267f, -0.00624645f,
    -0.01185068f, -0.03144731f, 0.03777607f, 0.00278718f, -0.02921014f,
    -0.01986694f, 0.03777607f, 0.56661550f, 0.28896755f, -0.04022174f,
    -0.01610267f, 0.00278718f, 0.28896755f, 0.14111091f, -0.03452303f,
    -0.00624645f, -0.02921014f, -0.04022174f, -0.03452303f, -0.01716200f,
};

void UpsampleImage(const ImageF& input, ImageF* output) {
  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();
  int64_t xsize2 = output->xsize();
  int64_t ysize2 = output->ysize();
  for (int64_t y = 0; y < ysize2; y++) {
    for (int64_t x = 0; x < xsize2; x++) {
      const auto* kernel = kernel00;
      if ((x & 1) && (y & 1)) {
        kernel = kernel11;
      } else if (x & 1) {
        kernel = kernel10;
      } else if (y & 1) {
        kernel = kernel01;
      }
      float sum = 0;
      int64_t x2 = x / 2;
      int64_t y2 = y / 2;
      float min_value = std::numeric_limits<float>::max();
      float max_value = std::numeric_limits<float>::lowest();

      for (int64_t ky = 0; ky < kSize; ky++) {
        for (int64_t kx = 0; kx < kSize; kx++) {
          int64_t xi = x2 - kSize / 2 + kx;
          int64_t yi = y2 - kSize / 2 + ky;
          if (xi < 0) xi = 0;
          if (xi >= xsize) xi = input.xsize() - 1;
          if (yi < 0) yi = 0;
          if (yi >= ysize) yi = input.ysize() - 1;
          min_value = std::min(min_value, input.Row(yi)[xi]);
          max_value = std::max(max_value, input.Row(yi)[xi]);
        }
      }

      for (int64_t ky = 0; ky < kSize; ky++) {
        for (int64_t kx = 0; kx < kSize; kx++) {
          int64_t xi = x2 - kSize / 2 + kx;
          int64_t yi = y2 - kSize / 2 + ky;
          if (xi < 0) xi = 0;
          if (xi >= xsize) xi = input.xsize() - 1;
          if (yi < 0) yi = 0;
          if (yi >= ysize) yi = input.ysize() - 1;
          sum += input.Row(yi)[xi] * kernel[ky * kSize + kx];
        }
      }
      output->Row(y)[x] = std::min(max_value, std::max(min_value, sum));
    }
  }
}

float UpsamplerDeriv(int64_t x2, int64_t y2, int64_t x, int64_t y) {
  const auto* kernel = kernel00;
  if ((x & 1) && (y & 1)) {
    kernel = kernel11;
  } else if (x & 1) {
    kernel = kernel10;
  } else if (y & 1) {
    kernel = kernel01;
  }
  int64_t ix = x / 2;
  int64_t iy = y / 2;
  int64_t kx = x2 - ix + kSize / 2;
  int64_t ky = y2 - iy + kSize / 2;
  if (kx < 0 || kx >= kSize || ky < 0 || ky >= kSize) return 0;
  return kernel[ky * kSize + kx];
}

void AntiUpsample(const ImageF& input, ImageF* d) {
  int64_t xsize = input.xsize();
  int64_t ysize = input.ysize();
  int64_t xsize2 = d->xsize();
  int64_t ysize2 = d->ysize();
  int64_t k0 = kSize - 1;
  int64_t k1 = kSize;
  for (int64_t y2 = 0; y2 < ysize2; ++y2) {
    auto* row = d->Row(y2);
    for (int64_t x2 = 0; x2 < xsize2; ++x2) {
      int64_t x0 = x2 * 2 - k0;
      if (x0 < 0) x0 = 0;
      int64_t x1 = x2 * 2 + k1 + 1;
      if (x1 > xsize) x1 = xsize;
      int64_t y0 = y2 * 2 - k0;
      if (y0 < 0) y0 = 0;
      int64_t y1 = y2 * 2 + k1 + 1;
      if (y1 > ysize) y1 = ysize;
      float sum = 0;
      for (int64_t y = y0; y < y1; ++y) {
        const auto* row_in = input.Row(y);
        for (int64_t x = x0; x < x1; ++x) {
          double deriv = UpsamplerDeriv(x2, y2, x, y);
          sum += deriv * row_in[x];
        }
      }
      row[x2] = sum;
    }
  }
}

template <typename T>
Status ElwiseMul(const Plane<T>& image1, const Plane<T>& image2,
                Plane<T>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JXL_ENSURE(xsize == out->xsize());
  JXL_ENSURE(ysize == out->ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) row_out[x] = row1[x] * row2[x];
  }
  return true;
}

template <typename T>
Status ElwiseDiv(const Plane<T>& image1, const Plane<T>& image2,
                Plane<T>* out) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JXL_ENSURE(xsize == out->xsize());
  JXL_ENSURE(ysize == out->ysize());
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out->Row(y);
    for (size_t x = 0; x < xsize; ++x) row_out[x] = row1[x] / row2[x];
  }
  return true;
}

void ReduceRinging(const ImageF& initial, const ImageF& mask, ImageF& down) {
  int64_t xsize2 = down.xsize();
  int64_t ysize2 = down.ysize();
  for (size_t y = 0; y < down.ysize(); y++) {
    const float* row_mask = mask.Row(y);
    float* row_out = down.Row(y);
    for (size_t x = 0; x < down.xsize(); x++) {
      float v = row_out[x];
      float min_value = initial.Row(y)[x];
      float max_value = initial.Row(y)[x];
      for (int64_t yi = -1; yi < 2; yi++) {
        for (int64_t xi = -1; xi < 2; xi++) {
          int64_t x2 = static_cast<int64_t>(x) + xi;
          int64_t y2 = static_cast<int64_t>(y) + yi;
          if (x2 < 0 || y2 < 0 || x2 >= xsize2 || y2 >= ysize2) continue;
          min_value = std::min(min_value, initial.Row(y2)[x2]);
          max_value = std::max(max_value, initial.Row(y2)[x2]);
        }
      }
      constexpr float mask_multiplier = 2.0f;
      const float a = row_mask[x] * mask_multiplier;
      row_out[x] = std::min(max_value + a, std::max(min_value - a, v));
    }
  }
}

Status DownsampleImage2_Iterative(const ImageF& orig, ImageF* output) {
  int64_t xsize = orig.xsize();
  int64_t ysize = orig.ysize();
  int64_t xsize2 = DivCeil(orig.xsize(), 2);
  int64_t ysize2 = DivCeil(orig.ysize(), 2);
  JxlMemoryManager* memory_manager = orig.memory_manager();

  JXL_ASSIGN_OR_RETURN(ImageF box_downsample,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_RETURN_IF_ERROR(CopyImageTo(orig, &box_downsample));
  JXL_ASSIGN_OR_RETURN(box_downsample, DownsampleImage(box_downsample, 2));
  JXL_ASSIGN_OR_RETURN(ImageF mask,
                       ImageF::Create(memory_manager, box_downsample.xsize(),
                                      box_downsample.ysize()));
  CreateMask(box_downsample, mask);

  JXL_RETURN_IF_ERROR(output->ShrinkTo(xsize2, ysize2));
  JXL_ASSIGN_OR_RETURN(
      ImageF initial,
      ImageF::Create(memory_manager, DivCeil(orig.xsize(), 2) + kBlockDim,
                     DivCeil(orig.ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(initial.ShrinkTo(initial.xsize() - kBlockDim,
                                       initial.ysize() - kBlockDim));
  JXL_RETURN_IF_ERROR(DownsampleImage2_Sharper(orig, &initial));

  JXL_ASSIGN_OR_RETURN(
      ImageF down,
      ImageF::Create(memory_manager, initial.xsize(), initial.ysize()));
  JXL_RETURN_IF_ERROR(CopyImageTo(initial, &down));
  JXL_ASSIGN_OR_RETURN(ImageF up, ImageF::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(ImageF corr,
                       ImageF::Create(memory_manager, xsize, ysize));
  JXL_ASSIGN_OR_RETURN(ImageF corr2,
                       ImageF::Create(memory_manager, xsize2, ysize2));

  JXL_ASSIGN_OR_RETURN(ImageF weights,
                       ImageF::Create(memory_manager, xsize, ysize));
  for (size_t y = 0; y < weights.ysize(); y++) {
    auto* row = weights.Row(y);
    for (size_t x = 0; x < weights.xsize(); x++) row[x] = 1;
  }
  JXL_ASSIGN_OR_RETURN(ImageF weights2,
                       ImageF::Create(memory_manager, xsize2, ysize2));
  AntiUpsample(weights, &weights2);

  const size_t num_it = 3;
  for (size_t it = 0; it < num_it; ++it) {
    UpsampleImage(down, &up);
    JXL_ASSIGN_OR_RETURN(corr, LinComb<float>(1, orig, -1, up));
    JXL_RETURN_IF_ERROR(ElwiseMul(corr, weights, &corr));
    AntiUpsample(corr, &corr2);
    JXL_RETURN_IF_ERROR(ElwiseDiv(corr2, weights2, &corr2));
    JXL_ASSIGN_OR_RETURN(down, LinComb<float>(1, down, 1, corr2));
  }

  ReduceRinging(initial, mask, down);
  for (size_t y = 0; y < down.ysize(); y++) {
    for (size_t x = 0; x < down.xsize(); x++) output->Row(y)[x] = down.Row(y)[x];
  }
  return true;
}

}  // namespace

Status DownsampleImage2_Iterative(Image3F* opsin) {
  JxlMemoryManager* memory_manager = opsin->memory_manager();
  JXL_ASSIGN_OR_RETURN(
      Image3F downsampled,
      Image3F::Create(memory_manager, DivCeil(opsin->xsize(), 2) + kBlockDim,
                      DivCeil(opsin->ysize(), 2) + kBlockDim));
  JXL_RETURN_IF_ERROR(downsampled.ShrinkTo(downsampled.xsize() - kBlockDim,
                                           downsampled.ysize() - kBlockDim));

  JXL_ASSIGN_OR_RETURN(Image3F rgb,
                       Image3F::Create(memory_manager, opsin->xsize(),
                                       opsin->ysize()));
  OpsinParams opsin_params;
  opsin_params.Init(kDefaultIntensityTarget);
  JXL_RETURN_IF_ERROR(
      OpsinToLinear(*opsin, Rect(rgb), nullptr, &rgb, opsin_params));

  JXL_ASSIGN_OR_RETURN(
      ImageF mask,
      ImageF::Create(memory_manager, opsin->xsize(), opsin->ysize()));
  ButteraugliParams butter_params;
  JXL_ASSIGN_OR_RETURN(std::unique_ptr<ButteraugliComparator> butter,
                       ButteraugliComparator::Make(rgb, butter_params));
  JXL_RETURN_IF_ERROR(butter->Mask(&mask));
  JXL_ASSIGN_OR_RETURN(
      ImageF mask_fuzzy,
      ImageF::Create(memory_manager, opsin->xsize(), opsin->ysize()));

  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        DownsampleImage2_Iterative(opsin->Plane(c), &downsampled.Plane(c)));
  }
  *opsin = std::move(downsampled);
  return true;
}

StatusOr<Image3F> ReconstructImage(
    const FrameHeader& orig_frame_header, const PassesSharedState& shared,
    const std::vector<std::unique_ptr<ACImage>>& coeffs, ThreadPool* pool) {
  const FrameDimensions& frame_dim = shared.frame_dim;
  JxlMemoryManager* memory_manager = shared.memory_manager;

  FrameHeader frame_header = orig_frame_header;
  frame_header.UpdateFlag(shared.image_features.patches.HasAny(),
                          FrameHeader::kPatches);
  frame_header.UpdateFlag(shared.image_features.splines.HasAny(),
                          FrameHeader::kSplines);
  frame_header.color_transform = ColorTransform::kNone;

  auto metadata = jxl::make_unique<CodecMetadata>();
  *metadata = *frame_header.nonserialized_metadata;
  metadata->m.extra_channel_info.clear();
  metadata->m.num_extra_channels = metadata->m.extra_channel_info.size();
  frame_header.nonserialized_metadata = metadata.get();
  frame_header.extra_channel_upsampling.clear();

  const bool is_gray = shared.metadata->m.color_encoding.IsGray();
  auto dec_state = jxl::make_unique<PassesDecoderState>(memory_manager);
  JXL_RETURN_IF_ERROR(
      dec_state->output_encoding_info.SetFromMetadata(*shared.metadata));
  JXL_RETURN_IF_ERROR(dec_state->output_encoding_info.MaybeSetColorEncoding(
      ColorEncoding::LinearSRGB(is_gray)));
  dec_state->shared = &shared;
  JXL_RETURN_IF_ERROR(dec_state->Init(frame_header));

  ImageBundle decoded(memory_manager, &shared.metadata->m);
  decoded.origin = frame_header.frame_origin;
  JXL_ASSIGN_OR_RETURN(
      Image3F tmp,
      Image3F::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
  JXL_RETURN_IF_ERROR(decoded.SetFromImage(
      std::move(tmp), dec_state->output_encoding_info.color_encoding));

  PassesDecoderState::PipelineOptions options;
  options.use_slow_render_pipeline = false;
  options.coalescing = false;
  options.render_spotcolors = false;
  options.render_noise = true;
  JXL_RETURN_IF_ERROR(dec_state->PreparePipeline(
      frame_header, &shared.metadata->m, &decoded, options));

  AlignedArray<GroupDecCache> group_dec_caches;
  const auto allocate_storage = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(
        dec_state->render_pipeline->PrepareForThreads(num_threads,
                                                       /*use_group_ids=*/false));
    JXL_ASSIGN_OR_RETURN(group_dec_caches,
                         AlignedArray<GroupDecCache>::Create(memory_manager,
                                                              num_threads));
    return true;
  };
  const auto process_group = [&](const uint32_t group_index,
                                 const size_t thread) -> Status {
    if (frame_header.loop_filter.epf_iters > 0) {
      JXL_RETURN_IF_ERROR(ComputeSigma(frame_header.loop_filter,
                                       frame_dim.BlockGroupRect(group_index),
                                       dec_state.get()));
    }
    RenderPipelineInput input =
        dec_state->render_pipeline->GetInputBuffers(group_index, thread);
    JXL_RETURN_IF_ERROR(DecodeGroupForRoundtrip(
        frame_header, coeffs, group_index, dec_state.get(),
        &group_dec_caches[thread], thread, input, nullptr, nullptr));
    if ((frame_header.flags & FrameHeader::kNoise) != 0) {
      PrepareNoiseInput(*dec_state, shared.frame_dim, frame_header, group_index,
                        thread);
    }
    JXL_RETURN_IF_ERROR(input.Done());
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, frame_dim.num_groups, allocate_storage,
                                process_group, "ReconstructImage"));
  return std::move(*decoded.color());
}

float ComputeBlockL2Distance(const Image3F& a, const Image3F& b,
                             const ImageF& mask1x1, size_t by, size_t bx) {
  Rect rect(bx * kBlockDim, by * kBlockDim, kBlockDim, kBlockDim, a.xsize(),
            a.ysize());
  float err2[3] = {0.0f};
  for (size_t y = 0; y < rect.ysize(); ++y) {
    const float* row_a[3] = {
        rect.ConstPlaneRow(a, 0, y), rect.ConstPlaneRow(a, 1, y),
        rect.ConstPlaneRow(a, 2, y)};
    const float* row_b[3] = {
        rect.ConstPlaneRow(b, 0, y), rect.ConstPlaneRow(b, 1, y),
        rect.ConstPlaneRow(b, 2, y)};
    const float* row_mask = rect.ConstRow(mask1x1, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      float mask_value = row_mask[x];
      float mask2 = mask_value * mask_value;
      for (int i = 0; i < 3; ++i) {
        float diff = row_a[i][x] - row_b[i][x];
        err2[i] += mask2 * diff * diff;
      }
    }
  }
  static const double kW[] = {12.339445295782363, 1.0, 0.2};
  return kW[0] * err2[0] + kW[1] * err2[1] + kW[2] * err2[2];
}

Status ComputeARHeuristics(const FrameHeader& frame_header,
                           PassesEncoderState* enc_state,
                           const Image3F& orig_opsin, const Rect& rect,
                           ThreadPool* pool) {
  const CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  const ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageB& epf_sharpness = shared.epf_sharpness;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  float clamped_butteraugli = std::min(5.0f, cparams.butteraugli_distance);
  if (cparams.butteraugli_distance < kMinButteraugliForDynamicAR ||
      cparams.speed_tier > SpeedTier::kWombat ||
      frame_header.loop_filter.epf_iters == 0) {
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    return true;
  }
  std::vector<uint8_t> epf_steps;
  if (cparams.butteraugli_distance > 7.0f) {
    // At very aggressive compression, include the strongest EPF candidate.
    // The later per-context penalty prevents it from taking over everywhere.
    epf_steps = {0, 4, 7};
  } else if (cparams.butteraugli_distance > 4.5f) {
    epf_steps = {0, 4, 7};
  } else {
    epf_steps = {0, 2, 7};
  }
  static const int kNumEPFVals = 8;
  size_t epf_steps_lut[kNumEPFVals] = {0};
  for (size_t i = 0; i < epf_steps.size(); ++i) epf_steps_lut[epf_steps[i]] = i;

  std::array<ImageF, kNumEPFVals> error_images;
  for (uint8_t val : epf_steps) {
    FillPlane(val, &epf_sharpness, Rect(epf_sharpness));
    JXL_ASSIGN_OR_RETURN(
        Image3F decoded,
        ReconstructImage(frame_header, shared, enc_state->coeffs, pool));
    JXL_ASSIGN_OR_RETURN(
        error_images[val],
        ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                       frame_dim.ysize_blocks));
    for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
      float* error_row = error_images[val].Row(by);
      for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
        error_row[bx] = ComputeBlockL2Distance(
            orig_opsin, decoded, initial_quant_masking1x1, by, bx);
      }
    }
  }

  std::vector<std::vector<size_t>> histo(9, std::vector<size_t>(kNumEPFVals));
  std::vector<size_t> totals(9, 1);
  const float lowq = std::min(
      1.0f, std::max(0.0f, (cparams.butteraugli_distance - 2.5f) / 4.5f));
  // At aggressive compression, EPF can erase the very texture that the
  // perceptual AQ deliberately preserved. Give the unfiltered candidate a
  // stronger RD preference while still allowing smoothing where measured
  // error warrants it.
  const float kFavorNoSmoothing = 0.985f - 0.035f * lowq;
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    uint8_t* JXL_RESTRICT out_row = epf_sharpness.Row(by);
    uint8_t* JXL_RESTRICT prev_row = epf_sharpness.Row(by > 0 ? by - 1 : 0);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      uint8_t best_val = 0;
      float best_error = std::numeric_limits<float>::max();
      uint8_t top_val = by > 0 ? prev_row[bx] : 0;
      uint8_t left_val = bx > 0 ? out_row[bx - 1] : 0;
      float top_error = error_images[top_val].Row(by)[bx];
      float left_error = error_images[left_val].Row(by)[bx];
      for (uint8_t val : epf_steps) {
        float error = error_images[val].Row(by)[bx];
        if (val == 0) error *= kFavorNoSmoothing;
        if (error < best_error) {
          best_val = val;
          best_error = error;
        }
      }
      if (best_error < std::min(top_error, left_error)) {
        out_row[bx] = best_val;
      } else if (top_error < left_error) {
        out_row[bx] = top_val;
      } else {
        out_row[bx] = left_val;
      }
      int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
      ++histo[context][out_row[bx]];
      ++totals[context];
    }
  }

  const float c3base = 0.98017198824148288f;
  const float c3clamp = 0.85970338919928291f;
  const float c3 = std::max(c3clamp, std::pow(c3base, clamped_butteraugli));
  static const float c5 = 0.1087690359555803f;
  float mul[3 * 3 * 3] = {0};
  for (uint8_t top_val : epf_steps) {
    for (uint8_t left_val : epf_steps) {
      for (uint8_t val : epf_steps) {
        int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
        const auto& ctx_histo = histo[context];
        const int mulix = epf_steps_lut[val] + 3 * context;
        mul[mulix] =
            1.0f / (1.0f + c5 *
                               std::log1p(static_cast<float>(ctx_histo[val]) /
                                          static_cast<float>(totals[context])) /
                               clamped_butteraugli);
        if (val == 0) mul[mulix] *= c3;
      }
    }
  }
  for (size_t by = 0; by < frame_dim.ysize_blocks; by++) {
    uint8_t* JXL_RESTRICT out_row = epf_sharpness.Row(by);
    uint8_t* JXL_RESTRICT prev_row = epf_sharpness.Row(by > 0 ? by - 1 : 0);
    for (size_t bx = 0; bx < frame_dim.xsize_blocks; bx++) {
      uint8_t best_val = 0;
      float best_error = std::numeric_limits<float>::max();
      uint8_t top_val = by > 0 ? prev_row[bx] : 0;
      uint8_t left_val = bx > 0 ? out_row[bx - 1] : 0;
      int context = epf_steps_lut[top_val] * 3 + epf_steps_lut[left_val];
      for (uint8_t val : epf_steps) {
        int mulix = epf_steps_lut[val] + 3 * context;
        float error = error_images[val].Row(by)[bx] * mul[mulix];
        if (error < best_error) {
          best_val = val;
          best_error = error;
        }
      }
      out_row[bx] = best_val;
    }
  }
  return true;
}

Status LossyFrameHeuristics(const FrameHeader& frame_header,
                            PassesEncoderState* enc_state,
                            ModularFrameEncoder* modular_frame_encoder,
                            const Image3F* linear, Image3F* opsin,
                            const Rect& rect, const JxlCmsInterface& cms,
                            ThreadPool* pool, AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  const bool streaming_mode = enc_state->streaming_mode;
  const bool initialize_global_state = enc_state->initialize_global_state;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = shared.frame_dim;
  ImageFeatures& image_features = shared.image_features;
  DequantMatrices& matrices = shared.matrices;
  Quantizer& quantizer = shared.quantizer;
  ImageF& initial_quant_masking1x1 = enc_state->initial_quant_masking1x1;
  ImageI& raw_quant_field = shared.raw_quant_field;
  ColorCorrelationMap& cmap = shared.cmap;
  AcStrategyImage& ac_strategy = shared.ac_strategy;
  BlockCtxMap& block_ctx_map = shared.block_ctx_map;
  JxlMemoryManager* memory_manager = enc_state->memory_manager();

  bool override_splines = cparams.custom_splines.HasAny();
  if (override_splines) image_features.splines.SetData(cparams.custom_splines);
  if (!streaming_mode && cparams.speed_tier <= SpeedTier::kSquirrel) {
    if (!override_splines) image_features.splines = FindSplines(*opsin);
    JXL_RETURN_IF_ERROR(image_features.splines.InitializeDrawCache(
        opsin->xsize(), opsin->ysize(), cmap.base()));
    image_features.splines.SubtractFrom(opsin);
  }

  if (!streaming_mode &&
      ApplyOverride(cparams.patches,
                    cparams.speed_tier <= SpeedTier::kSquirrel)) {
    JXL_RETURN_IF_ERROR(
        FindBestPatchDictionary(*opsin, enc_state, cms, pool, aux_out));
    JXL_RETURN_IF_ERROR(
        PatchDictionaryEncoder::SubtractFrom(image_features.patches, opsin));
  }

  const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);

  AcStrategyHeuristics acs_heuristics(memory_manager, cparams);
  CfLHeuristics cfl_heuristics(memory_manager);
  ImageF initial_quant_field;
  ImageF initial_quant_masking;

  if (cparams.speed_tier > SpeedTier::kHare ||
      cparams.disable_perceptual_optimizations) {
    JXL_ASSIGN_OR_RETURN(initial_quant_field,
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    JXL_ASSIGN_OR_RETURN(initial_quant_masking,
                         ImageF::Create(memory_manager, frame_dim.xsize_blocks,
                                        frame_dim.ysize_blocks));
    float q = 0.79f / cparams.butteraugli_distance;
    FillImage(q, &initial_quant_field);
    float masking = 1.0f / (q + 0.001f);
    FillImage(masking, &initial_quant_masking);
    if (cparams.disable_perceptual_optimizations) {
      JXL_ASSIGN_OR_RETURN(
          initial_quant_masking1x1,
          ImageF::Create(memory_manager, frame_dim.xsize, frame_dim.ysize));
      FillImage(masking, &initial_quant_masking1x1);
    }
    quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);
  } else {
    float butteraugli_distance_for_iqf = cparams.butteraugli_distance;
    if (!frame_header.loop_filter.gab) {
      butteraugli_distance_for_iqf *= 0.62f;
    }
    JXL_ASSIGN_OR_RETURN(
        initial_quant_field,
        InitialQuantField(butteraugli_distance_for_iqf, *opsin, rect, pool,
                          1.0f, &initial_quant_masking,
                          &initial_quant_masking1x1));
    float q = 0.39f / cparams.butteraugli_distance;
    quantizer.ComputeGlobalScaleAndQuant(quant_dc, q, 0);
  }

  if (frame_header.loop_filter.gab) {
    // At high distance, deliberately under-compensate the decoder-side
    // Gaborish slightly. This trades a small amount of ultimate sharpness for
    // substantially less ringing/broad halos around hard edges.
    const float distance = cparams.butteraugli_distance;
    const float t = std::min(1.0f,
                             std::max(0.0f, (distance - 2.0f) / 10.0f));
    const float gab_weight = 1.0f - 0.10f * t;
    float weight[3] = {gab_weight, gab_weight, gab_weight};
    JXL_RETURN_IF_ERROR(GaborishInverse(opsin, rect, weight, pool));
  }

  if (initialize_global_state) {
    JXL_RETURN_IF_ERROR(FindBestDequantMatrices(
        memory_manager, cparams, modular_frame_encoder, &matrices));
  }

  JXL_RETURN_IF_ERROR(cfl_heuristics.Init(rect));
  JXL_RETURN_IF_ERROR(acs_heuristics.Init(*opsin, rect, initial_quant_field,
                                          initial_quant_masking,
                                          initial_quant_masking1x1, &matrices));

  auto process_tile = [&](const uint32_t tid, const size_t thread) -> Status {
    size_t n_enc_tiles =
        DivCeil(frame_dim.xsize_blocks, kEncTileDimInBlocks);
    size_t tx = tid % n_enc_tiles;
    size_t ty = tid / n_enc_tiles;
    size_t by0 = ty * kEncTileDimInBlocks;
    size_t by1 =
        std::min((ty + 1) * kEncTileDimInBlocks, frame_dim.ysize_blocks);
    size_t bx0 = tx * kEncTileDimInBlocks;
    size_t bx1 =
        std::min((tx + 1) * kEncTileDimInBlocks, frame_dim.xsize_blocks);
    Rect r(bx0, by0, bx1 - bx0, by1 - by0);

    if (cparams.speed_tier <= SpeedTier::kSquirrel) {
      JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
          r, *opsin, rect, matrices,
          /*ac_strategy=*/nullptr,
          /*raw_quant_field=*/nullptr,
          /*quantizer=*/nullptr, /*fast=*/false, thread, &cmap));
    }

    JXL_RETURN_IF_ERROR(
        acs_heuristics.ProcessRect(r, cmap, &ac_strategy, thread));

    JXL_RETURN_IF_ERROR(AdjustQuantField(
        ac_strategy, r, cparams.butteraugli_distance, &initial_quant_field));
    quantizer.SetQuantFieldRect(initial_quant_field, r, &raw_quant_field);

    if (cparams.speed_tier <= SpeedTier::kHare) {
      JXL_RETURN_IF_ERROR(cfl_heuristics.ComputeTile(
          r, *opsin, rect, matrices, &ac_strategy, &raw_quant_field, &quantizer,
          /*fast=*/cparams.speed_tier >= SpeedTier::kWombat, thread, &cmap));
    }
    return true;
  };
  size_t num_tiles =
      DivCeil(frame_dim.xsize_blocks, kEncTileDimInBlocks) *
      DivCeil(frame_dim.ysize_blocks, kEncTileDimInBlocks);
  const auto prepare = [&](const size_t num_threads) -> Status {
    JXL_RETURN_IF_ERROR(acs_heuristics.PrepareForThreads(num_threads));
    JXL_RETURN_IF_ERROR(cfl_heuristics.PrepareForThreads(num_threads));
    return true;
  };
  JXL_RETURN_IF_ERROR(
      RunOnPool(pool, 0, num_tiles, prepare, process_tile, "Enc Heuristics"));

  JXL_RETURN_IF_ERROR(acs_heuristics.Finalize(frame_dim, ac_strategy, aux_out));

  if (!streaming_mode && !cparams.disable_perceptual_optimizations) {
    ImageB& epf_sharpness = shared.epf_sharpness;
    FillPlane(static_cast<uint8_t>(4), &epf_sharpness, Rect(epf_sharpness));
    JXL_RETURN_IF_ERROR(FindBestQuantizer(frame_header, linear, *opsin,
                                          initial_quant_field, enc_state, cms,
                                          pool, aux_out));
  }

  if (cparams.speed_tier < SpeedTier::kFalcon && initialize_global_state) {
    FindBestBlockEntropyModel(cparams, raw_quant_field, ac_strategy,
                              &block_ctx_map);
  }
  return true;
}

}  // namespace jxl
