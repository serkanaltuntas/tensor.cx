// Guarded-kernel intrinsics use the same RNE/gradual-underflow implementation
// as primitives. Stable sigmoid/SiLU follow the experimental DSL contract,
// independently of ordinary tensor SiLU's CPU-compatible formula.
uint tensorcx_add(uint a, uint b) { return cx_f32_add_bits(a, b); }
uint tensorcx_sub(uint a, uint b) { return nn_sub(a, b); }
uint tensorcx_mul(uint a, uint b) { return cx_f32_mul_bits(a, b); }
uint tensorcx_div(uint a, uint b) { return cx_f32_div_bits(a, b); }
uint tensorcx_sqrt(uint a) { return math_sqrt(a); }
uint tensorcx_abs(uint a) { return a & 0x7fffffffu; }
uint tensorcx_exp(uint a) { return nn_exp(a); }
uint tensorcx_log(uint a) { return math_log(a); }
uint tensorcx_rsqrt(uint a) { return cx_f32_div_bits(NN_ONE, math_sqrt(a)); }
uint tensorcx_tanh(uint a) { return nn_tanh(a); }
uint tensorcx_sigmoid(uint a) {
    if (nn_nan(a)) return a | 0x00400000u;
    uint e = nn_exp((a & 0x7fffffffu) | CX_F32_SIGN);
    return cx_f32_div_bits((a & CX_F32_SIGN) != 0u ? e : NN_ONE,
                          cx_f32_add_bits(NN_ONE, e));
}
uint tensorcx_silu(uint a) {
    if (nn_nan(a)) return a | 0x00400000u;
    uint magnitude = a & 0x7fffffffu;
    uint half_exp = nn_exp(cx_f32_mul_bits(magnitude | CX_F32_SIGN, 0x3f000000u));
    uint denominator = cx_f32_add_bits(NN_ONE, cx_f32_mul_bits(half_exp, half_exp));
    uint bounded = (nn_less(0x43800000u, magnitude) ? 0x43800000u : magnitude) | CX_F32_SIGN;
    uint tail = cx_f32_mul_bits(cx_f32_mul_bits(bounded, half_exp), half_exp);
    return cx_f32_div_bits((a & CX_F32_SIGN) != 0u ? tail : a, denominator);
}
uint tensorcx_minimum(uint a, uint b) {
    if (nn_nan(a)) return a | 0x00400000u;
    if (nn_nan(b)) return b | 0x00400000u;
    if (((a | b) & 0x7fffffffu) == 0u) return (a | b) & CX_F32_SIGN;
    return nn_less(a, b) ? a : b;
}
uint tensorcx_maximum(uint a, uint b) {
    if (nn_nan(a)) return a | 0x00400000u;
    if (nn_nan(b)) return b | 0x00400000u;
    if (((a | b) & 0x7fffffffu) == 0u) return (a & b) & CX_F32_SIGN;
    return nn_less(b, a) ? a : b;
}
bool tensorcx_lt(uint a, uint b) { return !nn_nan(a) && !nn_nan(b) && nn_less(a, b); }
bool tensorcx_lte(uint a, uint b) { return tensorcx_lt(a, b) || nn_equal(a, b); }
bool tensorcx_gt(uint a, uint b) { return tensorcx_lt(b, a); }
bool tensorcx_gte(uint a, uint b) { return tensorcx_lte(b, a); }
bool tensorcx_eq(uint a, uint b) { return nn_equal(a, b); }
bool tensorcx_neq(uint a, uint b) { return !nn_equal(a, b); }
