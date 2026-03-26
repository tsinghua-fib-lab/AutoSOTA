#include "common.h"

template<class scalar_t>
void hogwild_attention_cpu(float* out, float* att_out, float* qk_out, float scale,
                           const int* locations, const scalar_t* queries,
                           const int* fragment_lengths,
                           const scalar_t** key_fragments,
                           const scalar_t** value_fragments,
                           Shape shape) {
    // Input:   keys: [Hkv, fragment_lengths[i], E] for i in [F]
    //          values: [Hkv, fragment_lengths[i], Ev] for i in [F]
    //          fragment_lengths: [F]
    //          queries: [F, W, Hq, S, E]
    //          locations [F, W, S]
    // Output:  [W, Hq, S, Ev]
    // attention mask: s attends to l iff locations[b, s] >= l (i.e., shifted causal masking)

    int Ev = shape.Ev;
    int W = shape.W;
    int Hq = shape.Hq;
    int S = shape.S;
    int E = shape.E;

    std::vector<float> scratch(Ev, 0.0);
    std::ptrdiff_t q_stride = E * S * Hq * W;

    int total = 0;
    for (int f = 0; f < shape.F; ++f) {
        total += fragment_lengths[f];
    }

    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < Hq; ++h) {
            int hkv = h * shape.Hkv / Hq;
            for (int s = 0; s < S; ++s) {
                std::ptrdiff_t q_offset = ((w * Hq + h) * S + s) * E;

                // determine maximum
                float maximum = std::numeric_limits<float>::lowest();
                for (int f = 0; f < shape.F; ++f) {
                    int q_loc = locations[(f * W + w) * S + s];
                    int L = fragment_lengths[f];
                    int maxL = std::min(L, q_loc + 1);
                    for (int l = 0; l < maxL; ++l) {
                        std::ptrdiff_t kv_offset = (hkv * L + l) * E;
                        float qk = 0;
                        for (int e = 0; e < E; ++e) {
                            qk += (float)queries[f * q_stride + q_offset + e] * (float)key_fragments[f][kv_offset + e];
                        }
                        if (qk > maximum) {
                            maximum = qk;
                        }
                    }
                }

                // determine logsumexp
                float lse = 0;
                for (int f = 0; f < shape.F; ++f) {
                    int q_loc = locations[(f * W + w) * S + s];
                    int L = fragment_lengths[f];
                    int maxL = std::min(L, q_loc + 1);
                    for (int l = 0; l < maxL; ++l) {
                        std::ptrdiff_t kv_offset = (hkv * L + l) * E;
                        float qk = 0;
                        for (int e = 0; e < E; ++e) {
                            qk += (float)queries[f * q_stride + q_offset + e] * (float)key_fragments[f][kv_offset + e];
                        }
                        lse += std::exp(scale * (qk - maximum));
                    }
                }

                // combine values
                std::fill_n(scratch.begin(), Ev, 0.f);
                int pos_offset = 0;
                for (int f = 0; f < shape.F; ++f) {
                    int q_loc = locations[(f * W + w) * S + s];
                    int L = fragment_lengths[f];
                    int maxL = std::min(L, q_loc + 1);
                    for (int l = 0; l < maxL; ++l) {
                        std::ptrdiff_t kv_offset = (hkv * L + l) * E;
                        float qk = 0;
                        for (int e = 0; e < E; ++e) {
                            qk += (float)queries[f * q_stride + q_offset + e] * (float)key_fragments[f][kv_offset + e];
                        }
                        float att = std::exp(scale * (qk - maximum)) / lse;
                        if (att_out) {
                            att_out[(w * Hq * S + h * S + s) * total + pos_offset] = att;
                            qk_out[(w * Hq * S + h * S + s) * total + pos_offset] = qk;
                            ++pos_offset;
                        }
                        kv_offset = (hkv * L + l) * Ev;
                        for (int e = 0; e < Ev; ++e) {
                            scratch[e] += att * (float)value_fragments[f][kv_offset + e];
                        }
                    }
                    for (int l = maxL; l < L; ++l) {
                        if (att_out) {
                            att_out[(w * Hq * S + h * S + s) * total + pos_offset] = 0;
                            qk_out[(w * Hq * S + h * S + s) * total + pos_offset] = 0;
                            ++pos_offset;
                        }
                    }
                }

                // write result
                for (int e = 0; e < Ev; ++e) {
                    out[((w * Hq + h) * S + s) * Ev + e] = scratch[e];
                }
            }
        }
    }
}
