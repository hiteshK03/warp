// SPDX-FileCopyrightText: Copyright (c) 2026 The Warp-Metal Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Double-single ("df64") emulation of Warp's float64 for the Metal backend.
//
// Apple GPUs have no double type: MSL rejects `double` outright ("'double' is
// not supported in Metal"). Kernels that use wp.float64 / vec3d / mat33d / quatd
// therefore cannot compile natively. This header supplies wp::float64 as a
// two-float (hi, lo) pair using Dekker/Knuth error-free transformations, so the
// generic templated math (mat_t/vec_t/inverse/eig3/...) compiles against it
// unchanged.
//
// Precision: ~2x the float32 mantissa (roughly 44-48 significant bits) instead
// of binary64's 53. That is enough for the small-matrix conditioning work the
// emulation was built for (e.g. Newton's inertia eigen-decomposition and
// inverse, which use fp64 intermediates to avoid float32 determinant underflow
// and reciprocal overflow).
//
// RANGE IS NOT WIDENED: both components are float32, so the exponent range is
// float32's. Values beyond ~3.4e38 overflow, and subnormal-scale work can lose
// the low word. This is NOT a drop-in IEEE binary64. Code needing true binary64
// range must stay off Metal.
//
// Cost: several float ops per arithmetic op (two-sum is 6 flops, two-product
// with FMA is 2). Expect roughly 5-15x a native float op. Fine for per-body
// 3x3 setup kernels; not intended for hot inner loops.

#include <metal_stdlib>

// WP_THREAD comes from builtin.h, which is included after crt.h; this header is
// pulled in from metal_crt.h, so provide the qualifier directly.
#ifndef WP_THREAD
#define WP_THREAD thread
#endif

namespace wp {

// ---- error-free transformations (Dekker / Knuth) ----

// Exact sum: a + b = s + e, with s = fl(a+b). Knuth two-sum, branch-free.
inline void df_two_sum(float a, float b, thread float& s, thread float& e)
{
    s = a + b;
    float bb = s - a;
    float err = (a - (s - bb)) + (b - bb);
    e = err;
}

// Exact product: a * b = p + e, using FMA for the residual.
inline void df_two_prod(float a, float b, thread float& p, thread float& e)
{
    p = a * b;
    e = metal::fma(a, b, -p);
}

struct float64 {
    float hi;
    float lo;

    // Default-initialize to zero so `float64 x;` and value-init agree.
    inline float64() : hi(0.0f), lo(0.0f) { }
    inline float64(float v) : hi(v), lo(0.0f) { }
    inline float64(float h, float l) : hi(h), lo(l) { }

    // Construct from a literal double at *compile time* on the host side of the
    // shader: MSL has no double, so codegen emits float literals; this overload
    // exists for integral conversions used by generic code (e.g. T(0), T(1)).
    inline float64(int v) : hi(float(v)), lo(0.0f) { }
    inline float64(unsigned v) : hi(float(v)), lo(0.0f) { }

    inline explicit operator float() const { return hi + lo; }
    inline explicit operator int() const { return int(hi + lo); }
    inline explicit operator unsigned() const { return unsigned(hi + lo); }
    inline explicit operator bool() const { return (hi + lo) != 0.0f; }
};

// Renormalize: make |lo| <= ulp(hi)/2 so the pair stays canonical.
inline float64 df_normalize(float h, float l)
{
    float s, e;
    df_two_sum(h, l, s, e);
    return float64(s, e);
}

// ---- arithmetic ----

inline float64 operator+(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    float s, e;
    df_two_sum(a.hi, b.hi, s, e);
    e += a.lo + b.lo;
    return df_normalize(s, e);
}

inline float64 operator-(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    float s, e;
    df_two_sum(a.hi, -b.hi, s, e);
    e += a.lo - b.lo;
    return df_normalize(s, e);
}

inline float64 operator-(const float64 WP_THREAD& a)
{
    return float64(-a.hi, -a.lo);
}

inline float64 operator*(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    float p, e;
    df_two_prod(a.hi, b.hi, p, e);
    e += a.hi * b.lo + a.lo * b.hi;
    return df_normalize(p, e);
}

inline float64 operator/(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    // One Newton refinement of the reciprocal of the high word.
    float q1 = a.hi / b.hi;
    // r = a - q1*b (computed in df64 to keep the residual accurate)
    float64 r = a - float64(q1) * b;
    float q2 = r.hi / b.hi;
    return df_normalize(q1, q2);
}

inline float64 WP_THREAD& operator+=(float64 WP_THREAD& a, const float64 WP_THREAD& b) { a = a + b; return a; }
inline float64 WP_THREAD& operator-=(float64 WP_THREAD& a, const float64 WP_THREAD& b) { a = a - b; return a; }
inline float64 WP_THREAD& operator*=(float64 WP_THREAD& a, const float64 WP_THREAD& b) { a = a * b; return a; }
inline float64 WP_THREAD& operator/=(float64 WP_THREAD& a, const float64 WP_THREAD& b) { a = a / b; return a; }

// ---- comparisons (compare the canonical sum) ----

inline bool operator==(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return (a.hi + a.lo) == (b.hi + b.lo); }
inline bool operator!=(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return !(a == b); }
inline bool operator<(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool operator>(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return b < a; }
inline bool operator<=(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return !(b < a); }
inline bool operator>=(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return !(a < b); }

// ---- math ----

inline float64 abs(const float64 WP_THREAD& a)
{
    return a.hi < 0.0f ? -a : a;
}

inline float64 sqrt(const float64 WP_THREAD& a)
{
    if (a.hi == 0.0f)
        return float64(0.0f);
    float s = metal::sqrt(a.hi);
    // residual: (a - s*s) / (2*s), with s*s computed exactly via FMA
    float p, e;
    df_two_prod(s, s, p, e);
    float64 r = a - float64(p, e);
    return df_normalize(s, (r.hi / (2.0f * s)));
}


inline bool isnan(const float64 WP_THREAD& a) { return metal::isnan(a.hi) || metal::isnan(a.lo); }
inline bool isinf(const float64 WP_THREAD& a) { return metal::isinf(a.hi); }
inline bool isfinite(const float64 WP_THREAD& a) { return metal::isfinite(a.hi) && metal::isfinite(a.lo); }

inline float64 floor(const float64 WP_THREAD& a) { return float64(metal::floor(a.hi + a.lo)); }
inline float64 ceil(const float64 WP_THREAD& a) { return float64(metal::ceil(a.hi + a.lo)); }
inline float64 rint(const float64 WP_THREAD& a) { return float64(metal::rint(a.hi + a.lo)); }

// Reciprocal / division helpers used by Warp's approx_* paths.
inline float64 approx_rcp(const float64 WP_THREAD& a) { return float64(1.0f) / a; }
inline float64 approx_div(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return a / b; }

}  // namespace wp

// ---- named entry points used by Warp codegen and DECLARE_FLOAT_OPS ----
// Codegen emits wp::add/mul/... rather than operators, and DECLARE_FLOAT_OPS
// (enabled for float64 under WP_METAL_DF64) routes min/max through these.
namespace wp {

inline float64 _wp_native_fmin(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    // NaN-propagating semantics matching the float/double overloads in builtin.h
    return (a <= b) ? a : ((b == b) ? b : a);
}
inline float64 _wp_native_fmax(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    return (a >= b) ? a : ((b == b) ? b : a);
}


inline float64 mod(const float64 WP_THREAD& a, const float64 WP_THREAD& b)
{
    return a - b * floor(a / b);
}
inline float64 floordiv(const float64 WP_THREAD& a, const float64 WP_THREAD& b) { return floor(a / b); }
inline float64 copysign(const float64 WP_THREAD& x, const float64 WP_THREAD& y)
{
    float64 ax = abs(x);
    return y < float64(0.0f) ? -ax : ax;
}

// Conversions between df64 and the native scalar types. Warp codegen emits
// wp::float64(x) / wp::float32(x) style casts.
inline float64 cast_float64(float x) { return float64(x); }
inline float cast_float32(const float64 WP_THREAD& x) { return float(x); }

}  // namespace wp

// ---- adjoints for the df64 math functions ----
// Warp emits adj_* for every builtin used in a differentiable kernel.
namespace wp {

// Signature matches DECLARE_ADJOINTS in builtin.h: (x, ret, adj_x, adj_ret).
inline void adj_sqrt(
    const float64 WP_THREAD& x,
    const float64 WP_THREAD& ret,
    float64 WP_THREAD& adj_x,
    const float64 WP_THREAD& adj_ret)
{
    // d/dx sqrt(x) = 1 / (2*sqrt(x)); reuse the forward result.
    if (ret != float64(0.0f))
        adj_x += float64(0.5f) * (float64(1.0f) / ret) * adj_ret;
}

inline void adj_fabs(const float64 WP_THREAD& x, const float64 WP_THREAD& ret, float64 WP_THREAD& adj_x, const float64 WP_THREAD& adj_ret)
{
    adj_x += (x < float64(0.0f)) ? -adj_ret : adj_ret;
}

inline void adj_floor(const float64 WP_THREAD& x, const float64 WP_THREAD& ret, float64 WP_THREAD& adj_x, const float64 WP_THREAD& adj_ret) { }
inline void adj_ceil(const float64 WP_THREAD& x, const float64 WP_THREAD& ret, float64 WP_THREAD& adj_x, const float64 WP_THREAD& adj_ret) { }
inline void adj_rint(const float64 WP_THREAD& x, const float64 WP_THREAD& ret, float64 WP_THREAD& adj_x, const float64 WP_THREAD& adj_ret) { }

inline void adj_approx_div(
    const float64 WP_THREAD& a,
    const float64 WP_THREAD& b,
    const float64 WP_THREAD& ret,
    float64 WP_THREAD& adj_a,
    float64 WP_THREAD& adj_b,
    const float64 WP_THREAD& adj_ret)
{
    adj_a += approx_div(adj_ret, b);
    adj_b -= approx_div(float64(adj_ret * ret), b);
}

}  // namespace wp
