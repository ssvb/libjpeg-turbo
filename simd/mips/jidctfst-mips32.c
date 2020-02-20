/*
 * MIPS DSP and DSPr2 optimizations for libjpeg-turbo
 *
 * Copyright (C) 2020, Siarhei Siamashka. All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define JPEG_INTERNALS
#include "../../jinclude.h"
#include "../../jpeglib.h"
#include "../../jsimd.h"
#include "../../jdct.h"
#include "../../jsimddct.h"
#include "../jsimd.h"

typedef short v2i16 __attribute__ ((vector_size(4)));
typedef signed char v4i8 __attribute__ ((vector_size(4)));

/**********************************************************************/
/* Emulate the missing DSPr2 instructions on DSPr1 hardware           */
/**********************************************************************/

#if (__mips_dsp_rev < 2)
static inline v2i16 __builtin_mips_mul_ph(v2i16 a, v2i16 b)
{
  return a * b; /* GCC can take care of it */
}

static inline v2i16 __builtin_mips_mulq_s_ph(v2i16 a, v2i16 b)
{
#ifdef USE_ACCURATE_ROUNDING
  return __builtin_mips_mulq_rs_ph(a, b);
#else
  return __builtin_mips_precrq_ph_w(__builtin_mips_muleq_s_w_phl(a, b),
                                    __builtin_mips_muleq_s_w_phr(a, b));
#endif
}

static inline v4i8 __builtin_mips_precr_qb_ph(v2i16 a, v2i16 b)
{
  return __builtin_mips_precrq_qb_ph(__builtin_mips_shll_ph(a, 8),
                                     __builtin_mips_shll_ph(b, 8));
}
#endif

/**********************************************************************/

static inline void save_transposed_2x2(void *upper_row, void *lower_row,
                                       v2i16 upper_val, v2i16 lower_val)
{
#ifdef __MIPSEL__
  ((int16_t *)upper_row)[0] = (int16_t)(int32_t)lower_val;
  ((int16_t *)upper_row)[1] = (int16_t)(int32_t)upper_val;
  *(v2i16 *)lower_row = __builtin_mips_precrq_ph_w((int32_t)upper_val,
                                                   (int32_t)lower_val);
#else
  *(v2i16 *)upper_row = __builtin_mips_precrq_ph_w((int32_t)upper_val,
                                                   (int32_t)lower_val);
  ((int16_t *)lower_row)[0] = (int16_t)(int32_t)upper_val;
  ((int16_t *)lower_row)[1] = (int16_t)(int32_t)lower_val;
#endif
}

/**********************************************************************/

#if (__mips_dsp_rev < 2)
void jsimd_idct_ifast_dsp(void *dct_table_, JCOEFPTR coef_block,
                            JSAMPARRAY output_buf, JDIMENSION output_col)
#else
void jsimd_idct_ifast_dspr2(void *dct_table_, JCOEFPTR coef_block,
                            JSAMPARRAY output_buf, JDIMENSION output_col)
#endif
{
  v2i16 workspace[32];
  int i;
  v2i16 *dest;
  v2i16 row0, row1, row2, row3, row4, row5, row6, row7;
  v2i16 q0, q1, q2, q3, q4, q5, q6, q7;
  v2i16 *inptr = (v2i16 *)coef_block;
  v2i16 *wsptr = (v2i16 *)workspace;
  v2i16 *qptr = (v2i16 *)dct_table_;

  v2i16 XFIX_1_082392200 = {277 * 128 - 256 * 128, 277 * 128 - 256 * 128};
  v2i16 XFIX_1_414213562 = {362 * 128 - 256 * 128, 362 * 128 - 256 * 128};
  v2i16 XFIX_1_847759065 = {473 * 128 - 256 * 128, 473 * 128 - 256 * 128};
  v2i16 XFIX_2_613125930 = {669 * 128 - 512 * 128, 669 * 128 - 512 * 128};

  /* Pass 1: process columns */

  for (i = 0; i < 4; i++) {
    /* load two columns */
    row0 = inptr[4 * 0];
    row1 = inptr[4 * 1];
    row2 = inptr[4 * 2];
    row3 = inptr[4 * 3];
    row4 = inptr[4 * 4];
    row5 = inptr[4 * 5];
    row6 = inptr[4 * 6];
    row7 = inptr[4 * 7];

    if (((int32_t)row1 | (int32_t)row2 | (int32_t)row3 | (int32_t)row4 |
        (int32_t)row5 | (int32_t)row6 | (int32_t)row7) == 0) {
      /* special case, rows 1-7 are all zero */
      q0 = __builtin_mips_mul_ph(row0, qptr[4 * 0]);
#ifdef __MIPSEL__
      q2 = __builtin_mips_precrq_ph_w((int32_t)q0, (int32_t)q0);
      q1 = __builtin_mips_repl_ph((int32_t)q0);
#else
      q1 = __builtin_mips_precrq_ph_w((int32_t)q0, (int32_t)q0);
      q2 = __builtin_mips_repl_ph((int32_t)q0);
#endif
      wsptr[0 + 0] = q1;
      wsptr[0 + 1] = q1;
      wsptr[0 + 2] = q1;
      wsptr[0 + 3] = q1;
      wsptr[4 + 0] = q2;
      wsptr[4 + 1] = q2;
      wsptr[4 + 2] = q2;
      wsptr[4 + 3] = q2;

      inptr += 1;
      qptr  += 1;
      wsptr += 8;
      continue;
    }

    /* dequantize */
    row0 = __builtin_mips_mul_ph(row0, qptr[4 * 0]);
    row1 = __builtin_mips_mul_ph(row1, qptr[4 * 1]);
    row2 = __builtin_mips_mul_ph(row2, qptr[4 * 2]);
    row3 = __builtin_mips_mul_ph(row3, qptr[4 * 3]);
    row4 = __builtin_mips_mul_ph(row4, qptr[4 * 4]);
    row5 = __builtin_mips_mul_ph(row5, qptr[4 * 5]);
    row6 = __builtin_mips_mul_ph(row6, qptr[4 * 6]);
    row7 = __builtin_mips_mul_ph(row7, qptr[4 * 7]);

    /* 1-D IDCT kernel (borrowed from ARM NEON code) */
    q2   = __builtin_mips_subq_ph(row2, row6);
    row6 = __builtin_mips_addq_ph(row2, row6);
    q1   = __builtin_mips_subq_ph(row3, row5);
    row5 = __builtin_mips_addq_ph(row3, row5);
    q5   = __builtin_mips_subq_ph(row1, row7);
    row7 = __builtin_mips_addq_ph(row1, row7);
    q4   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q6   = __builtin_mips_mulq_s_ph(q1, XFIX_2_613125930);
    q3   = __builtin_mips_addq_ph(q1, q1);
    q1   = __builtin_mips_subq_ph(q5, q1);
    row2 = __builtin_mips_addq_ph(q2, q4);
    q4   = __builtin_mips_mulq_s_ph(q1, XFIX_1_847759065);
    q2   = __builtin_mips_subq_ph(row7, row5);
    q3   = __builtin_mips_addq_ph(q3, q6);
    q6   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q1   = __builtin_mips_addq_ph(q1, q4);
    q4   = __builtin_mips_mulq_s_ph(q5, XFIX_1_082392200);
    row2 = __builtin_mips_subq_ph(row2, row6);
    q2   = __builtin_mips_addq_ph(q2, q6);
    q6   = __builtin_mips_subq_ph(row0, row4);
    row4 = __builtin_mips_addq_ph(row0, row4);
    row1 = __builtin_mips_addq_ph(q5, q4);
    q5   = __builtin_mips_addq_ph(q6, row2);
    row2 = __builtin_mips_subq_ph(q6, row2);
    q6   = __builtin_mips_addq_ph(row7, row5);
    row0 = __builtin_mips_addq_ph(row4, row6);
    q3   = __builtin_mips_subq_ph(q6, q3);
    row4 = __builtin_mips_subq_ph(row4, row6);
    q3   = __builtin_mips_subq_ph(q3, q1);
    q1   = __builtin_mips_subq_ph(row1, q1);
    q2   = __builtin_mips_addq_ph(q3, q2);
    row7 = __builtin_mips_subq_ph(row0, q6);
    q1   = __builtin_mips_addq_ph(q1, q2);
    row0 = __builtin_mips_addq_ph(row0, q6);
    row6 = __builtin_mips_addq_ph(q5, q3);
    row1 = __builtin_mips_subq_ph(q5, q3);
    row5 = __builtin_mips_subq_ph(row2, q2);
    row2 = __builtin_mips_addq_ph(row2, q2);
    row3 = __builtin_mips_subq_ph(row4, q1);
    row4 = __builtin_mips_addq_ph(row4, q1);

    /* transpose them and store as two rows */
    save_transposed_2x2(wsptr + 0 + 0, wsptr + 4 + 0, row0, row1);
    save_transposed_2x2(wsptr + 0 + 1, wsptr + 4 + 1, row2, row3);
    save_transposed_2x2(wsptr + 0 + 2, wsptr + 4 + 2, row4, row5);
    save_transposed_2x2(wsptr + 0 + 3, wsptr + 4 + 3, row6, row7);

    inptr += 1;
    qptr  += 1;
    wsptr += 8;
  }

  wsptr = (v2i16 *)workspace;

  /* Pass 2: process rows */

  for (i = 0; i < 4; i++) {
    /* load two columns */
    row0 = wsptr[4 * 0];
    row1 = wsptr[4 * 1];
    row2 = wsptr[4 * 2];
    row3 = wsptr[4 * 3];
    row4 = wsptr[4 * 4];
    row5 = wsptr[4 * 5];
    row6 = wsptr[4 * 6];
    row7 = wsptr[4 * 7];

    /* 1-D IDCT kernel (borrowed from ARM NEON code) */
    q2   = __builtin_mips_subq_ph(row2, row6);
    row6 = __builtin_mips_addq_ph(row2, row6);
    q1   = __builtin_mips_subq_ph(row3, row5);
    row5 = __builtin_mips_addq_ph(row3, row5);
    row0 = __builtin_mips_addq_ph(row0, (v2i16)0x00800080 << 5); /* +0x80 */
    q5   = __builtin_mips_subq_ph(row1, row7);
    row7 = __builtin_mips_addq_ph(row1, row7);
    q4   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q6   = __builtin_mips_mulq_s_ph(q1, XFIX_2_613125930);
    q3   = __builtin_mips_addq_ph(q1, q1);
    q1   = __builtin_mips_subq_ph(q5, q1);
    row2 = __builtin_mips_addq_ph(q2, q4);
    q4   = __builtin_mips_mulq_s_ph(q1, XFIX_1_847759065);
    q2   = __builtin_mips_subq_ph(row7, row5);
    q3   = __builtin_mips_addq_ph(q3, q6);
    q6   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q1   = __builtin_mips_addq_ph(q1, q4);
    q4   = __builtin_mips_mulq_s_ph(q5, XFIX_1_082392200);
    row2 = __builtin_mips_subq_ph(row2, row6);
    q2   = __builtin_mips_addq_ph(q2, q6);
    q6   = __builtin_mips_subq_ph(row0, row4);
    row4 = __builtin_mips_addq_ph(row0, row4);
    row1 = __builtin_mips_addq_ph(q5, q4);
    q5   = __builtin_mips_addq_ph(q6, row2);
    row2 = __builtin_mips_subq_ph(q6, row2);
    q6   = __builtin_mips_addq_ph(row7, row5);
    row0 = __builtin_mips_addq_ph(row4, row6);
    q3   = __builtin_mips_subq_ph(q6, q3);
    row4 = __builtin_mips_subq_ph(row4, row6);
    q3   = __builtin_mips_subq_ph(q3, q1);
    q1   = __builtin_mips_subq_ph(row1, q1);
    q2   = __builtin_mips_addq_ph(q3, q2);
    row7 = __builtin_mips_subq_ph(row0, q6);
    q1   = __builtin_mips_addq_ph(q1, q2);
    row0 = __builtin_mips_addq_ph(row0, q6);
    row6 = __builtin_mips_addq_ph(q5, q3);
    row1 = __builtin_mips_subq_ph(q5, q3);
    row5 = __builtin_mips_subq_ph(row2, q2);
    row2 = __builtin_mips_addq_ph(row2, q2);
    row3 = __builtin_mips_subq_ph(row4, q1);
    row4 = __builtin_mips_addq_ph(row4, q1);

    /* shift into the right position for a fancy PRECRU_S_QB.PH instruction */
    row0 = __builtin_mips_shll_s_ph(row0, 2);
    row1 = __builtin_mips_shll_s_ph(row1, 2);
    row2 = __builtin_mips_shll_s_ph(row2, 2);
    row3 = __builtin_mips_shll_s_ph(row3, 2);
    row4 = __builtin_mips_shll_s_ph(row4, 2);
    row5 = __builtin_mips_shll_s_ph(row5, 2);
    row6 = __builtin_mips_shll_s_ph(row6, 2);
    row7 = __builtin_mips_shll_s_ph(row7, 2);

    /* pack into unsigned bytes with saturation (range limit to 0-255) */
#ifdef __MIPSEL__
    q0 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row3, row2);
    q1 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row1, row0);
    q2 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row7, row6);
    q3 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row5, row4);
#else
    q0 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row0, row1);
    q1 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row2, row3);
    q2 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row4, row5);
    q3 = (v2i16)__builtin_mips_precrqu_s_qb_ph(row6, row7);
#endif

    q4 = (v2i16)__builtin_mips_precrq_qb_ph(q0, q1);
    q5 = (v2i16)__builtin_mips_precrq_qb_ph(q2, q3);
    q6 = (v2i16)__builtin_mips_precr_qb_ph(q0, q1);
    q7 = (v2i16)__builtin_mips_precr_qb_ph(q2, q3);

    /* store to the destination */
    dest = (v2i16 *)(*output_buf++ + output_col);
    *dest++ = q4;
    *dest++ = q5;

    dest = (v2i16 *)(*output_buf++ + output_col);
    *dest++ = q6;
    *dest++ = q7;

    wsptr += 1;
  }
}
