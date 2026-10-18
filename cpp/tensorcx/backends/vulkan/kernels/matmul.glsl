// One invocation owns one output, accumulating in CPU row-major K order.
// Integer binary32 helpers round multiply and add separately, preserving
// gradual underflow without optional float controls or fused operations.
void matmul(uint index) {
    uint k = p.scalar_bits;
    if (k == 0u) { output_data[index] = 0u; return; }
    uint n = p.scalar_left, m = p.integer_dtype;
    uint col = index % n, row = (index / n) % m;
    uint batch = (index / n) / m;
    uint left = row * k, right = col;
    for (uint axis = p.rank; axis > 0u; --axis) {
        uint coordinate = batch % metadata[axis - 1u];
        batch /= metadata[axis - 1u];
        left += coordinate * metadata[p.rank + axis - 1u];
        right += coordinate * metadata[2u * p.rank + axis - 1u];
    }
    uint sum = 0u;
    for (uint inner = 0u; inner < k; ++inner)
        sum = cx_f32_add_bits(sum, cx_f32_mul_bits(lhs[left + inner], rhs[right + inner * n]));
    output_data[index] = sum;
}
