//! SIMD-accelerated vector and matrix primitives.
//!
//! Uses `core::arch::x86_64` SSE/AVX on x86_64 targets; scalar fallback on all others.
//! All operations are `const`-friendly where possible and parameterized by dimension.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Add two arrays element-wise, returning the result.
pub fn simd_add<const N: usize>(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    let mut out = [0.0f32; N];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { avx_add(a, b, &mut out) };
            return out;
        }
        if is_x86_feature_detected!("sse") {
            unsafe { sse_add(a, b, &mut out) };
            return out;
        }
    }

    scalar_add(a, b, &mut out);
    out
}

/// Element-wise subtraction of two arrays.
pub fn simd_sub<const N: usize>(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    let mut out = [0.0f32; N];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { avx_sub(a, b, &mut out) };
            return out;
        }
        if is_x86_feature_detected!("sse") {
            unsafe { sse_sub(a, b, &mut out) };
            return out;
        }
    }

    scalar_sub(a, b, &mut out);
    out
}

/// Element-wise multiply of two arrays.
pub fn simd_mul<const N: usize>(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    let mut out = [0.0f32; N];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { avx_mul(a, b, &mut out) };
            return out;
        }
        if is_x86_feature_detected!("sse") {
            unsafe { sse_mul(a, b, &mut out) };
            return out;
        }
    }

    scalar_mul(a, b, &mut out);
    out
}

/// Dot product of two arrays.
pub fn simd_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { avx_dot(a, b) };
        }
        if is_x86_feature_detected!("sse") {
            return unsafe { sse_dot(a, b) };
        }
    }

    scalar_dot(a, b)
}

/// Scale an array by a scalar.
pub fn simd_scale<const N: usize>(a: &[f32; N], s: f32) -> [f32; N] {
    let mut out = [0.0f32; N];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { avx_scale(a, s, &mut out) };
            return out;
        }
        if is_x86_feature_detected!("sse") {
            unsafe { sse_scale(a, s, &mut out) };
            return out;
        }
    }

    scalar_scale(a, s, &mut out);
    out
}

/// L2 norm squared.
pub fn simd_norm2<const N: usize>(a: &[f32; N]) -> f32 {
    simd_dot(a, a)
}

/// L2 norm.
pub fn simd_norm<const N: usize>(a: &[f32; N]) -> f32 {
    simd_norm2(a).sqrt()
}

// ---------------------------------------------------------------------------
// Scalar fallbacks
// ---------------------------------------------------------------------------

fn scalar_add<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    for i in 0..N {
        out[i] = a[i] + b[i];
    }
}

fn scalar_sub<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    for i in 0..N {
        out[i] = a[i] - b[i];
    }
}

fn scalar_mul<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    for i in 0..N {
        out[i] = a[i] * b[i];
    }
}

fn scalar_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..N {
        acc += a[i] * b[i];
    }
    acc
}

fn scalar_scale<const N: usize>(a: &[f32; N], s: f32, out: &mut [f32; N]) {
    for i in 0..N {
        out[i] = a[i] * s;
    }
}

// ---------------------------------------------------------------------------
// SSE implementations (4-wide)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
unsafe fn sse_add<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    let mut i = 0;
    while i + 4 <= N {
        let va = unsafe { _mm_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm_loadu_ps(b.as_ptr().add(i)) };
        let vr = unsafe { _mm_add_ps(va, vb) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 4;
    }
    while i < N {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn sse_sub<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    let mut i = 0;
    while i + 4 <= N {
        let va = unsafe { _mm_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm_loadu_ps(b.as_ptr().add(i)) };
        let vr = unsafe { _mm_sub_ps(va, vb) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 4;
    }
    while i < N {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn sse_mul<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    let mut i = 0;
    while i + 4 <= N {
        let va = unsafe { _mm_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm_loadu_ps(b.as_ptr().add(i)) };
        let vr = unsafe { _mm_mul_ps(va, vb) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 4;
    }
    while i < N {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn sse_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut acc = unsafe { _mm_setzero_ps() };
    let mut i = 0;
    while i + 4 <= N {
        let va = unsafe { _mm_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm_loadu_ps(b.as_ptr().add(i)) };
        acc = unsafe { _mm_add_ps(acc, _mm_mul_ps(va, vb)) };
        i += 4;
    }
    let mut arr = [0.0f32; 4];
    unsafe { _mm_storeu_ps(arr.as_mut_ptr(), acc) };
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];
    while i < N {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn sse_scale<const N: usize>(a: &[f32; N], s: f32, out: &mut [f32; N]) {
    let vs = unsafe { _mm_set1_ps(s) };
    let mut i = 0;
    while i + 4 <= N {
        let va = unsafe { _mm_loadu_ps(a.as_ptr().add(i)) };
        let vr = unsafe { _mm_mul_ps(va, vs) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 4;
    }
    while i < N {
        out[i] = a[i] * s;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// AVX implementations (8-wide)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
unsafe fn avx_add<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    let mut i = 0;
    while i + 8 <= N {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let vr = unsafe { _mm256_add_ps(va, vb) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 8;
    }
    while i < N {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx_sub<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    let mut i = 0;
    while i + 8 <= N {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let vr = unsafe { _mm256_sub_ps(va, vb) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 8;
    }
    while i < N {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx_mul<const N: usize>(a: &[f32; N], b: &[f32; N], out: &mut [f32; N]) {
    let mut i = 0;
    while i + 8 <= N {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let vr = unsafe { _mm256_mul_ps(va, vb) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 8;
    }
    while i < N {
        out[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx_dot<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut acc = unsafe { _mm256_setzero_ps() };
    let mut i = 0;
    while i + 8 <= N {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        acc = unsafe { _mm256_add_ps(acc, _mm256_mul_ps(va, vb)) };
        i += 8;
    }
    let mut arr = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(arr.as_mut_ptr(), acc) };
    let mut sum = arr.iter().sum();
    while i < N {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx_scale<const N: usize>(a: &[f32; N], s: f32, out: &mut [f32; N]) {
    let vs = unsafe { _mm256_set1_ps(s) };
    let mut i = 0;
    while i + 8 <= N {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vr = unsafe { _mm256_mul_ps(va, vs) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(i), vr) };
        i += 8;
    }
    while i < N {
        out[i] = a[i] * s;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c = simd_add(&a, &b);
        assert_eq!(c, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_simd_mul_8() {
        let a = [2.0f32; 8];
        let b = [3.0f32; 8];
        let c = simd_mul(&a, &b);
        assert_eq!(c, [6.0f32; 8]);
    }

    #[test]
    fn test_simd_dot_8() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert!((simd_dot(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_scale_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = simd_scale(&a, 2.0);
        assert_eq!(c, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_simd_norm() {
        let a = [3.0f32, 4.0];
        assert!((simd_norm(&a) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_odd_dimension() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let c = simd_add(&a, &b);
        assert_eq!(c, [5.0, 7.0, 9.0]);
        assert!((simd_dot(&a, &b) - 32.0).abs() < 1e-6);
    }
}
