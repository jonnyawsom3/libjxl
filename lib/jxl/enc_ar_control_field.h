// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_AR_CONTROL_FIELD_H_
#define LIB_JXL_ENC_AR_CONTROL_FIELD_H_

#include <stddef.h>

#include <vector>

#include <jxl/memory_manager.h>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"

namespace jxl {

struct PassesEncoderState;

// Fast, local estimate of the EPF detail-preservation control field. This
// avoids reconstructing the entire encoded image several times just to select
// a per-block sharpness value.
struct ArControlFieldHeuristics {
  struct TempImages {
    Status InitOnce(JxlMemoryManager* memory_manager) {
      if (laplacian_sqrsum.xsize() != 0) return true;
      JXL_ASSIGN_OR_RETURN(
          laplacian_sqrsum,
          ImageF::Create(memory_manager, kEncTileDim + 4, kEncTileDim + 4));
      JXL_ASSIGN_OR_RETURN(
          sqrsum_00,
          ImageF::Create(memory_manager, kEncTileDim / 4, kEncTileDim / 4));
      JXL_ASSIGN_OR_RETURN(
          sqrsum_22,
          ImageF::Create(memory_manager, kEncTileDim / 4 + 1,
                         kEncTileDim / 4 + 1));
      return true;
    }

    ImageF laplacian_sqrsum;
    ImageF sqrsum_00;
    ImageF sqrsum_22;
  };

  Status PrepareForThreads(size_t num_threads) {
    temp_images.resize(num_threads);
    return true;
  }

  Status RunRect(const Rect& block_rect, const Image3F& opsin,
                 const ImageF& initial_quant_field,
                 const FrameHeader& frame_header,
                 PassesEncoderState* enc_state, size_t thread);

  std::vector<TempImages> temp_images;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_AR_CONTROL_FIELD_H_
