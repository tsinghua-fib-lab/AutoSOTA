#include <cstdio>

template<class scalar_t>
void rope_cpu(scalar_t* rotated_queries, const scalar_t* queries, const float* cosines, const float* sines,
                           int F, int W, int Hq, int S, int E) {
    // Input:   queries: [W, Hq, S, E]
    //          sines, cosines: [F, W, S, E]
    // Output:  [F, W, Hq, S, E]
    for(int f = 0; f < F; ++f) {
        for(int w = 0; w < W; ++w) {
            for(int h = 0; h < Hq; ++h) {
                for(int s = 0; s < S; ++s) {
                    const scalar_t* query = queries + ((w * Hq + h) * S + s) * E;
                    scalar_t* result = rotated_queries + (((f * W + w) * Hq + h) * S + s) * E;
                    int offset = (((f*W + w) * S + s) * E);
                    for (int e = 0; e < E / 2; e++) {
                        float x1 = query[e];
                        float x2 = query[e + E/2];
                        // fetch a tuple of activations, which we imagine as a complex number

                        result[e] = x1 * cosines[offset + e] - x2 * sines[offset + e];
                        result[e + E/2] = x2 * cosines[offset + e + E/2] + x1 * sines[offset + e + E/2];
                    }
                }
            }
        }
    }
}
