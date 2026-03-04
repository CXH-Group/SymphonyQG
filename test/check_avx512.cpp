#include <cstdio>
#include <immintrin.h>

int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    __m512 b = _mm512_set1_ps(2.0f);
    __m512 c = _mm512_add_ps(a, b);
    float result[16];
    _mm512_storeu_ps(result, c);
    printf("AVX-512 works: %f\n", result[0]);
    return 0;
}
