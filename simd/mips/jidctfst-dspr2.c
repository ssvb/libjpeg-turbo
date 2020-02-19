/*
 * MIPS DSPr2 optimizations for libjpeg-turbo
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

static inline v2i16 transpose_upper(v2i16 upper, v2i16 lower)
{
  return __builtin_mips_precrq_ph_w((int32_t)upper, (int32_t)lower);
}

static inline v2i16 transpose_lower(v2i16 upper, v2i16 lower)
{
  return __builtin_mips_precr_sra_ph_w((int32_t)upper, (int32_t)lower, 0);
}

#ifdef __MIPSEL__
# define UPPER_ROW 4
# define LOWER_ROW 0
#else
# define UPPER_ROW 0
# define LOWER_ROW 4
#endif

/**********************************************************************/

void jsimd_idct_ifast_dspr2(void *dct_table_, JCOEFPTR coef_block,
                            JSAMPARRAY output_buf, JDIMENSION output_col)
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
      q1 = transpose_upper(q0, q0);
      q2 = transpose_lower(q0, q0);

      wsptr[UPPER_ROW + 0] = q1;
      wsptr[UPPER_ROW + 1] = q1;
      wsptr[UPPER_ROW + 2] = q1;
      wsptr[UPPER_ROW + 3] = q1;
      wsptr[LOWER_ROW + 0] = q2;
      wsptr[LOWER_ROW + 1] = q2;
      wsptr[LOWER_ROW + 2] = q2;
      wsptr[LOWER_ROW + 3] = q2;

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
    q2   = __builtin_mips_subu_ph(row2, row6);
    row6 = __builtin_mips_addu_ph(row2, row6);
    q1   = __builtin_mips_subu_ph(row3, row5);
    row5 = __builtin_mips_addu_ph(row3, row5);
    q5   = __builtin_mips_subu_ph(row1, row7);
    row7 = __builtin_mips_addu_ph(row1, row7);
    q4   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q6   = __builtin_mips_mulq_s_ph(q1, XFIX_2_613125930);
    q3   = __builtin_mips_addu_ph(q1, q1);
    q1   = __builtin_mips_subu_ph(q5, q1);
    row2 = __builtin_mips_addu_ph(q2, q4);
    q4   = __builtin_mips_mulq_s_ph(q1, XFIX_1_847759065);
    q2   = __builtin_mips_subu_ph(row7, row5);
    q3   = __builtin_mips_addu_ph(q3, q6);
    q6   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q1   = __builtin_mips_addu_ph(q1, q4);
    q4   = __builtin_mips_mulq_s_ph(q5, XFIX_1_082392200);
    row2 = __builtin_mips_subu_ph(row2, row6);
    q2   = __builtin_mips_addu_ph(q2, q6);
    q6   = __builtin_mips_subu_ph(row0, row4);
    row4 = __builtin_mips_addu_ph(row0, row4);
    row1 = __builtin_mips_addu_ph(q5, q4);
    q5   = __builtin_mips_addu_ph(q6, row2);
    row2 = __builtin_mips_subu_ph(q6, row2);
    q6   = __builtin_mips_addu_ph(row7, row5);
    row0 = __builtin_mips_addu_ph(row4, row6);
    q3   = __builtin_mips_subu_ph(q6, q3);
    row4 = __builtin_mips_subu_ph(row4, row6);
    q3   = __builtin_mips_subu_ph(q3, q1);
    q1   = __builtin_mips_subu_ph(row1, q1);
    q2   = __builtin_mips_addu_ph(q3, q2);
    row7 = __builtin_mips_subu_ph(row0, q6);
    q1   = __builtin_mips_addu_ph(q1, q2);
    row0 = __builtin_mips_addu_ph(row0, q6);
    row6 = __builtin_mips_addu_ph(q5, q3);
    row1 = __builtin_mips_subu_ph(q5, q3);
    row5 = __builtin_mips_subu_ph(row2, q2);
    row2 = __builtin_mips_addu_ph(row2, q2);
    row3 = __builtin_mips_subu_ph(row4, q1);
    row4 = __builtin_mips_addu_ph(row4, q1);

    /* transpose them and store as two rows */
    wsptr[UPPER_ROW + 0] = transpose_upper(row0, row1);
    wsptr[UPPER_ROW + 1] = transpose_upper(row2, row3);
    wsptr[UPPER_ROW + 2] = transpose_upper(row4, row5);
    wsptr[UPPER_ROW + 3] = transpose_upper(row6, row7);
    wsptr[LOWER_ROW + 0] = transpose_lower(row0, row1);
    wsptr[LOWER_ROW + 1] = transpose_lower(row2, row3);
    wsptr[LOWER_ROW + 2] = transpose_lower(row4, row5);
    wsptr[LOWER_ROW + 3] = transpose_lower(row6, row7);

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
    q2   = __builtin_mips_subu_ph(row2, row6);
    row6 = __builtin_mips_addu_ph(row2, row6);
    q1   = __builtin_mips_subu_ph(row3, row5);
    row5 = __builtin_mips_addu_ph(row3, row5);
    q5   = __builtin_mips_subu_ph(row1, row7);
    row7 = __builtin_mips_addu_ph(row1, row7);
    q4   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q6   = __builtin_mips_mulq_s_ph(q1, XFIX_2_613125930);
    q3   = __builtin_mips_addu_ph(q1, q1);
    q1   = __builtin_mips_subu_ph(q5, q1);
    row2 = __builtin_mips_addu_ph(q2, q4);
    q4   = __builtin_mips_mulq_s_ph(q1, XFIX_1_847759065);
    q2   = __builtin_mips_subu_ph(row7, row5);
    q3   = __builtin_mips_addu_ph(q3, q6);
    q6   = __builtin_mips_mulq_s_ph(q2, XFIX_1_414213562);
    q1   = __builtin_mips_addu_ph(q1, q4);
    q4   = __builtin_mips_mulq_s_ph(q5, XFIX_1_082392200);
    row2 = __builtin_mips_subu_ph(row2, row6);
    q2   = __builtin_mips_addu_ph(q2, q6);
    q6   = __builtin_mips_subu_ph(row0, row4);
    row4 = __builtin_mips_addu_ph(row0, row4);
    row1 = __builtin_mips_addu_ph(q5, q4);
    q5   = __builtin_mips_addu_ph(q6, row2);
    row2 = __builtin_mips_subu_ph(q6, row2);
    q6   = __builtin_mips_addu_ph(row7, row5);
    row0 = __builtin_mips_addu_ph(row4, row6);
    q3   = __builtin_mips_subu_ph(q6, q3);
    row4 = __builtin_mips_subu_ph(row4, row6);
    q3   = __builtin_mips_subu_ph(q3, q1);
    q1   = __builtin_mips_subu_ph(row1, q1);
    q2   = __builtin_mips_addu_ph(q3, q2);
    row7 = __builtin_mips_subu_ph(row0, q6);
    q1   = __builtin_mips_addu_ph(q1, q2);
    row0 = __builtin_mips_addu_ph(row0, q6);
    row6 = __builtin_mips_addu_ph(q5, q3);
    row1 = __builtin_mips_subu_ph(q5, q3);
    row5 = __builtin_mips_subu_ph(row2, q2);
    row2 = __builtin_mips_addu_ph(row2, q2);
    row3 = __builtin_mips_subu_ph(row4, q1);
    row4 = __builtin_mips_addu_ph(row4, q1);

    /* range limit by using left shift with saturation */
    row0 = __builtin_mips_shll_s_ph(row0, 3);
    row1 = __builtin_mips_shll_s_ph(row1, 3);
    row2 = __builtin_mips_shll_s_ph(row2, 3);
    row3 = __builtin_mips_shll_s_ph(row3, 3);
    row4 = __builtin_mips_shll_s_ph(row4, 3);
    row5 = __builtin_mips_shll_s_ph(row5, 3);
    row6 = __builtin_mips_shll_s_ph(row6, 3);
    row7 = __builtin_mips_shll_s_ph(row7, 3);

    /* pack into bytes and add 0x80 to each of them */
#ifdef __MIPSEL__
    q0 = (v2i16)__builtin_mips_precrq_qb_ph(row3, row2);
    q1 = (v2i16)__builtin_mips_precrq_qb_ph(row1, row0);
    q4 = (v2i16)__builtin_mips_precrq_qb_ph(row7, row6);
    q5 = (v2i16)__builtin_mips_precrq_qb_ph(row5, row4);
#else
    q0 = (v2i16)__builtin_mips_precrq_qb_ph(row0, row1);
    q1 = (v2i16)__builtin_mips_precrq_qb_ph(row2, row3);
    q4 = (v2i16)__builtin_mips_precrq_qb_ph(row4, row5);
    q5 = (v2i16)__builtin_mips_precrq_qb_ph(row6, row7);
#endif

    q2 = (v2i16)__builtin_mips_precrq_qb_ph(q0, q1);
    q3 = (v2i16)__builtin_mips_precr_qb_ph(q0, q1);
    q2 = (v2i16)__builtin_mips_addu_qb((v4i8)q2, (v4i8)0x80808080);
    q3 = (v2i16)__builtin_mips_addu_qb((v4i8)q3, (v4i8)0x80808080);

    q6 = (v2i16)__builtin_mips_precrq_qb_ph(q4, q5);
    q7 = (v2i16)__builtin_mips_precr_qb_ph(q4, q5);
    q6 = (v2i16)__builtin_mips_addu_qb((v4i8)q6, (v4i8)0x80808080);
    q7 = (v2i16)__builtin_mips_addu_qb((v4i8)q7, (v4i8)0x80808080);

    /* store to the destination */
    dest = (v2i16 *)(*output_buf++ + output_col);
    *dest++ = q2;
    *dest++ = q6;

    dest = (v2i16 *)(*output_buf++ + output_col);
    *dest++ = q3;
    *dest++ = q7;

    wsptr += 1;
  }
}
