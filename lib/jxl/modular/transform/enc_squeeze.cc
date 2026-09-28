// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_squeeze.h"

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/squeeze.h"
#include "lib/jxl/modular/transform/squeeze_params.h"

namespace jxl {

namespace {

void DumpPGM(const Channel& ch, const char* direction, const char* kind,
             size_t pass, uint32_t c) {
  if (ch.w == 0 || ch.h == 0) return;

  char filename[256];
  snprintf(filename, sizeof(filename),
           "squeeze_%03zu_c%u_%s_%s.pgm",
           pass, c, direction, kind);

  pixel_type minv = ch.Row(0)[0];
  pixel_type maxv = minv;

  for (size_t y = 0; y < ch.h; ++y) {
    const pixel_type* row = ch.Row(y);
    for (size_t x = 0; x < ch.w; ++x) {
      minv = std::min(minv, row[x]);
      maxv = std::max(maxv, row[x]);
    }
  }

  FILE* f = fopen(filename, "wb");
  if (f == nullptr) return;

  fprintf(f, "P5\n%zu %zu\n255\n", ch.w, ch.h);

  const int64_t lo = static_cast<int64_t>(minv);
  const int64_t range = static_cast<int64_t>(maxv) - lo;

  for (size_t y = 0; y < ch.h; ++y) {
    const pixel_type* row = ch.Row(y);
    for (size_t x = 0; x < ch.w; ++x) {
      const uint8_t v =
          range == 0
              ? 0
              : static_cast<uint8_t>(
                    (static_cast<int64_t>(row[x]) - lo) * 255 / range);
      fwrite(&v, 1, 1, f);
    }
  }

  fclose(f);
}

}  // namespace

#define AVERAGE(X, Y) (((X) + (Y) + (((X) > (Y)) ? 1 : 0)) >> 1)

Status FwdHSqueeze(Image &input, int c, int rc) {
  const Channel &chin = input.channel[c];
  JxlMemoryManager *memory_manager = input.memory_manager();

  JXL_DEBUG_V(4, "Doing horizontal squeeze of channel %i to new channel %i",
              c, rc);

  JXL_ASSIGN_OR_RETURN(
      Channel chout,
      Channel::Create(memory_manager, (chin.w + 1) / 2, chin.h,
                      chin.hshift + 1, chin.vshift));
  JXL_ASSIGN_OR_RETURN(
      Channel chout_residual,
      Channel::Create(memory_manager, chin.w - chout.w,
                      chout.h, chin.hshift + 1, chin.vshift));
  chout.component = chin.component;
  chout_residual.component = chin.component;

  for (size_t y = 0; y < chout.h; y++) {
    const pixel_type *JXL_RESTRICT p_in = chin.Row(y);
    pixel_type *JXL_RESTRICT p_out = chout.Row(y);
    pixel_type *JXL_RESTRICT p_res = chout_residual.Row(y);
    for (size_t x = 0; x < chout_residual.w; x++) {
      pixel_type A = p_in[x * 2];
      pixel_type B = p_in[x * 2 + 1];
      pixel_type avg = AVERAGE(A, B);
      p_out[x] = avg;

      pixel_type diff = A - B;

      pixel_type next_avg = avg;
      if (x + 1 < chout_residual.w) {
        pixel_type C = p_in[x * 2 + 2];
        pixel_type D = p_in[x * 2 + 3];
        next_avg = AVERAGE(C, D);
      } else if (chin.w & 1) {
        next_avg = p_in[x * 2 + 2];
      }
      pixel_type left = (x > 0 ? p_in[x * 2 - 1] : avg);
      pixel_type tendency = SmoothTendency(left, avg, next_avg);

      p_res[x] = diff - tendency;
    }
    if (chin.w & 1) {
      int x = chout.w - 1;
      p_out[x] = p_in[x * 2];
    }
  }

  input.channel[c] = std::move(chout);
  input.channel.insert(input.channel.begin() + rc,
                       std::move(chout_residual));
  return true;
}

Status FwdVSqueeze(Image &input, int c, int rc) {
  const Channel &chin = input.channel[c];
  JxlMemoryManager *memory_manager = input.memory_manager();

  JXL_DEBUG_V(4, "Doing vertical squeeze of channel %i to new channel %i",
              c, rc);

  JXL_ASSIGN_OR_RETURN(
      Channel chout,
      Channel::Create(memory_manager, chin.w, (chin.h + 1) / 2,
                      chin.hshift, chin.vshift + 1));
  JXL_ASSIGN_OR_RETURN(
      Channel chout_residual,
      Channel::Create(memory_manager, chin.w, chin.h - chout.h,
                      chin.hshift, chin.vshift + 1));
  chout.component = chin.component;
  chout_residual.component = chin.component;

  ptrdiff_t onerow_in = chin.plane.PixelsPerRow();
  for (size_t y = 0; y < chout_residual.h; y++) {
    const pixel_type *JXL_RESTRICT p_in = chin.Row(y * 2);
    pixel_type *JXL_RESTRICT p_out = chout.Row(y);
    pixel_type *JXL_RESTRICT p_res = chout_residual.Row(y);
    for (size_t x = 0; x < chout.w; x++) {
      pixel_type A = p_in[x];
      pixel_type B = p_in[x + onerow_in];
      pixel_type avg = AVERAGE(A, B);
      p_out[x] = avg;

      pixel_type diff = A - B;

      pixel_type next_avg = avg;
      if (y + 1 < chout_residual.h) {
        pixel_type C = p_in[x + 2 * onerow_in];
        pixel_type D = p_in[x + 3 * onerow_in];
        next_avg = AVERAGE(C, D);
      } else if (chin.h & 1) {
        next_avg = p_in[x + 2 * onerow_in];
      }
      pixel_type top =
          (y > 0 ? p_in[static_cast<ptrdiff_t>(x) - onerow_in] : avg);
      pixel_type tendency = SmoothTendency(top, avg, next_avg);

      p_res[x] = diff - tendency;
    }
  }

  if (chin.h & 1) {
    size_t y = chout.h - 1;
    const pixel_type *p_in = chin.Row(y * 2);
    pixel_type *p_out = chout.Row(y);
    for (size_t x = 0; x < chout.w; x++) {
      p_out[x] = p_in[x];
    }
  }

  input.channel[c] = std::move(chout);
  input.channel.insert(input.channel.begin() + rc,
                       std::move(chout_residual));
  return true;
}

Status FwdSqueeze(Image &input, std::vector<SqueezeParams> parameters,
                  ThreadPool *pool) {
  static size_t dump_pass = 0;

  if (parameters.empty()) {
    DefaultSqueezeParameters(&parameters, input);
  }
  if (parameters.empty()) return false;

  for (auto &parameter : parameters) {
    const size_t pass = dump_pass++;

    JXL_RETURN_IF_ERROR(
        CheckMetaSqueezeParams(parameter, input.channel.size()));

    const bool horizontal = parameter.horizontal;
    const bool in_place = parameter.in_place;
    const uint32_t beginc = parameter.begin_c;
    const uint32_t endc = parameter.begin_c + parameter.num_c - 1;

    uint32_t offset;
    if (in_place) {
      offset = endc + 1;
    } else {
      offset = input.channel.size();
    }

    for (uint32_t c = beginc; c <= endc; c++) {
      const uint32_t rc = offset + c - beginc;

      if (horizontal) {
        JXL_RETURN_IF_ERROR(FwdHSqueeze(input, c, rc));
      } else {
        JXL_RETURN_IF_ERROR(FwdVSqueeze(input, c, rc));
      }

      DumpPGM(input.channel[c],
              horizontal ? "h" : "v",
              "avg", pass, c);
      DumpPGM(input.channel[rc],
              horizontal ? "h" : "v",
              "residual", pass, c);
    }
  }

  return true;
}

}  // namespace jxl
