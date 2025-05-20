/**
 * Fixed-Point Arithmetic Library
 * 
 * This library implements fixed-point arithmetic operations using a configurable
 * number of bits for the integer and fractional parts.
 */

#ifndef FIXED_POINT_LIB_H
#define FIXED_POINT_LIB_H

#include <stdint.h>
#include <math.h>

typedef int32_t fxp;

#define FXP_INT_BITS        (16)
#define FXP_FRAC_BITS       (16)

#define FXP_SCALE           (FXP_FRAC_BITS)
#define FXP_SCALE_FACTOR    (1 << FXP_SCALE)

#define FXP_FROM_INT(a)     (a << FXP_SCALE)
#define FXP_FROM_FLOAT(a)   ((fxp)((a) * FXP_SCALE_FACTOR))
#define FXP_TO_FLOAT(a)     ((float)(a) / FXP_SCALE_FACTOR)

#define FXP_ONE             (1 << FXP_SCALE)
#define FXP_ZERO            (0 << FXP_SCALE)

/* Fixed-point multiplication */
extern fxp fxp_mul(fxp a, fxp b);

/* Divide two fixed-point numbers */
extern fxp fxp_div(fxp a, fxp b);

/* Get the absolute value of a fixed-point number */
extern fxp fxp_abs(fxp a);

extern fxp fxptoa(char* str, fxp value);

#endif /* FIXED_POINT_LIB_H */