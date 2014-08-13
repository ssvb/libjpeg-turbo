#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define HAVE_RGB565 1

#define N      (5000)
#define WIDTH  ((1024 - 64 * 3) / 8 * 8)

double gettime (void)
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return (double)((int64_t)tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.;
}

void jsimd_ycc_extrgb565_convert_neon(int out_width,
                                      void *input_buf[], int input_row,
                                      void *output_buf[], int num_rows);

void jsimd_ycc_extxrgb_convert_neon(int out_width,
                                    void *input_buf[], int input_row,
                                    void *output_buf[], int num_rows);

void jsimd_ycc_extrgb_convert_neon(int out_width,
                                   void *input_buf[], int input_row,
                                   void *output_buf[], int num_rows);

int main(int argc, char *argv[])
{
    void (*jsimd_ycc_convert_neon)(int out_width,
                                   void *input_buf[], int input_row,
                                   void *output_buf[], int num_rows) = 0;
    int i, bpp = 0;
    int cpu_clock_speed = 0;

    char *rgb1 = memalign(16, WIDTH * 8);
    char *rgb2 = rgb1 + WIDTH * 4;

    char *y1 = rgb1;
    char *u1 = rgb1 + WIDTH;
    char *v1 = rgb1 + WIDTH * 2;

    char *y2 = rgb2;
    char *u2 = rgb2 + WIDTH;
    char *v2 = rgb2 + WIDTH * 2;

    void *input_buf_y[8] = { y1, y2, y1, y2, y1, y2, y1, y2 };
    void *input_buf_u[8] = { u1, u2, u1, u2, u1, u2, u1, u2 };
    void *input_buf_v[8] = { v1, v2, v1, v2, v1, v2, v1, v2 };
    void *input_buf[3]   = { input_buf_y, input_buf_u, input_buf_v };
    void *output_buf[8]  = { rgb2, rgb1, rgb2, rgb1, rgb2, rgb1, rgb2, rgb1 };

    memset(rgb1, 0x11, WIDTH * 4);
    memset(rgb2, 0x11, WIDTH * 4);

    if (argc > 1) {
        if (atoi(argv[1]) == 32) {
            bpp = 32;
            jsimd_ycc_convert_neon = jsimd_ycc_extxrgb_convert_neon;
        }
        if (atoi(argv[1]) == 24) {
            bpp = 24;
            jsimd_ycc_convert_neon = jsimd_ycc_extrgb_convert_neon;
        }
#ifdef HAVE_RGB565
        if (atoi(argv[1]) == 16) {
            bpp = 16;
            jsimd_ycc_convert_neon = jsimd_ycc_extrgb565_convert_neon;
        }
#endif
    }

    if (!jsimd_ycc_convert_neon) {
        printf("Usage: bench_yuv [rgb_bit_depth] [cpu_clock_frequency_in_mhz]\n");
        return 1;
    }

    if (argc > 2)
        cpu_clock_speed = atoi(argv[2]);

    double t1, t2;
    double min;
    int count = 0;

    do
    {
        t1 = gettime();
        for (i = 0; i < N; i++)
        {
            jsimd_ycc_convert_neon(WIDTH, input_buf, 0, output_buf, 8);
        }
        t2 = gettime();
        if (count == 1)
            min = t2 - t1;
        else if (t2 - t1 < min)
            min = t2 - t1;
        count++;
    } while (count < 20);

    printf("bpp: %d, speed: %.1f MPix/s (~%.1f cycles per 8 pixels at %d MHz)\n",
        bpp,
        (double)N * 8 * WIDTH / (min) / 1000000.,
        (min) * cpu_clock_speed * 1000000. * 8 / (N * 8 * WIDTH),
        cpu_clock_speed);
}
