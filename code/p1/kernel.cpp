// File       : kernel.cpp
// Description: Peak performance kernel implementation
// Copyright 2024 Harvard University. All Rights Reserved.

// TODO: implement a compute bound kernel to measure performance.  You are free
// to:
//      1. Change the function signature (return type and function parameter)
//         but you cannot change the name of the function.  When you change the
//         function signature, be sure to also update the kernel declaration in
//         the main.cpp file.
//      2. The kernel may not compute anything meaningful.  Your aim is to reach
//         as much performance (Gflop/s) as possible given your current
//         knowledge.  You can be creative here but make sure the compiler does
//         not optimize away code you want to keep in your kernel.
//      3. You may use any instructions (including AVX2 extensions) you like but
//         you are not allowed to write assembly directly or write inline
//         assembly in this source file.
//      4. You cannot use ISPC for the kernel implementation.

#include <immintrin.h> // Include for AVX2 intrinsics

void kernel() {
    // Declare arrays to store the data that AVX2 will process.
    // Using __m256 because it can hold 8 single-precision floats (256 bits total).
    __m256 vecA = _mm256_set1_ps(1.0f); // Initialize all elements to 1.0
    __m256 vecB = _mm256_set1_ps(2.0f); // Initialize all elements to 2.0
    __m256 result;

    // Perform a large number of operations to saturate the CPU's floating-point units
    for (int i = 0; i < 1000000; i++) {
        result = _mm256_add_ps(vecA, vecB); // Vector addition
        vecA = _mm256_mul_ps(result, vecB); // Vector multiplication, reusing the result
    }

    // Volatile to prevent the compiler from optimizing out the loop
    volatile __m256 sink = result;
}