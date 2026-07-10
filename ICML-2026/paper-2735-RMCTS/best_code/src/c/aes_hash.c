#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h> // immintrin.h subsumes below three
//#include <emmintrin.h>
//#include <wmmintrin.h>
//#include <smmintrin.h>

#define AES_ROUNDS 10

/*
Encrypts using AES-128 round function but with a constant key schedule.
The 128-bit CV is hard-coded (a random setting) as K0 and K1 below.
*/

#define K0 0x38262225f095512a
#define K1 0xba0f4c97ae6c10c6

/*
__m128i _mm_setr_epi64 (__m64, __m64) // emmintrin.h, SSE2 (CPUID)
__m128i _mm_aesenc_si128 (__m128i, __m128i) // wmmintrin.h, AES (CPUID)
__m128i _mm_xor_si128 (__m128i, __m128i) // emmintrin.h, SSE2 (CPUID)
__int64 _mm_extract_epi64 (__m128i, const int) // smmintrin.h, SSE4.1 (CPUID)
*/

uint64_t aes_hash(const char* const x, int64_t len_x) {
	int64_t i;
	int64_t num_chunks_256 = (len_x - 1ll)/32ll + 1ll; //32 bytes, 256 bits
	int64_t num_128i = 2ll * num_chunks_256;
	uint64_t b0,b1;
	__m128i B;
	__m128i K;
	int r;

	__m128i* X;
	assert(posix_memalign(((void **) (&X)), 32, 32*num_chunks_256) == 0);
	memset(X,0,32*num_chunks_256);
	memcpy(X,x,len_x * sizeof(char));
	K = _mm_setr_epi64((__m64) K0, (__m64) K1);
	for(r=0;r<AES_ROUNDS;r++) {
		X[0] = _mm_aesenc_si128(X[0],K);
		for(i=1ll;i<num_128i;i++) {
			//X[i] = _mm_xor_si128(X[i-1ll],X[i]); //removed for speed
			X[i] = _mm_aesenc_si128(X[i], K);
		}
	}
	B = _mm_setr_epi64((__m64) 0ull, (__m64) 0ull);
	for(i=0ll;i<num_128i;i++) {
		B = _mm_xor_si128(X[i], B);
	}
	free(X);
	b0 = (uint64_t) _mm_extract_epi64(B,0);
	b1 = (uint64_t) _mm_extract_epi64(B,1);
	return b0 ^ b1;
}
