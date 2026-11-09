"""
Algebra check for FLASH_TURING_FASTSM (runs on CPU, numpy only, no GPU needed).

Emulates the forward kernel's per-row online softmax in float32, visiting KV
blocks from last to first exactly like flash_fwd_kernel.h, in two variants:

  orig : S *= scale; mask; m = max(S); P = exp(S - m); alpha = exp(m_old - m);
         L = m + log(l)
  fast : mask raw S (-FLT_MAX); m = max(S_raw); P = exp2((S_raw - m) * scale*log2e);
         alpha = exp2((m_old - m) * scale*log2e); L = m*scale + log(l)
         (rows whose m is still -FLT_MAX keep L = -FLT_MAX, as in orig)

It checks (1) both match a float64 reference where the row has >=1 valid key,
(2) orig and fast agree even on fully-masked rows (causal, seqlen_q > seqlen_k),
(3) ragged seqlen_k (masking of padding columns) is handled the same.
This validates the math only. It says nothing about nvcc codegen or speed.
"""
import numpy as np

FLT_MAX = np.float32(np.finfo(np.float32).max)
LOG2E = np.float32(1.4426950408889634)


def run(q, k, v, scale, causal, bn, variant):
    sq, d = q.shape
    sk = k.shape[0]
    scale = np.float32(scale)
    c = np.float32(scale * LOG2E)
    nb = -(-sk // bn)
    out = np.zeros((sq, d), np.float32)
    L = np.zeros(sq, np.float32)
    for r in range(sq):
        m_old = FLT_MAX * np.float32(-1)
        l_old = np.float32(0)
        acc = np.zeros(d, np.float32)
        # causal bottom-right alignment like the kernel: col - row > sk - sq masked
        for nbk in range(nb - 1, -1, -1):
            cols = np.arange(nbk * bn, min((nbk + 1) * bn, sk))
            s = (q[r].astype(np.float32) @ k[cols].astype(np.float32).T).astype(np.float32)
            if variant == "orig":
                s = (s * scale).astype(np.float32)
            if causal:
                s = np.where(cols - r > sk - sq, -FLT_MAX, s)
            m = np.float32(max(m_old, s.max()))
            if variant == "orig":
                p = np.exp(s - m).astype(np.float32)
                alpha = np.exp(m_old - m).astype(np.float32)
            else:
                p = np.exp2(((s - m) * c).astype(np.float32)).astype(np.float32)
                alpha = np.exp2(((m_old - m) * c).astype(np.float32)).astype(np.float32)
            if m == -FLT_MAX:            # masking-step guard in the kernel: fully masked row -> P = 0
                p = np.zeros_like(p)
            l = np.float32(alpha * l_old + p.sum(dtype=np.float32))
            acc = (acc * alpha + (p @ v[cols].astype(np.float32))).astype(np.float32)
            m_old, l_old = m, l
        if l_old != 0:
            out[r] = acc / l_old
        if l_old == 0:
            L[r] = 0
        elif variant == "orig":
            L[r] = m_old + np.log(l_old)
        else:
            L[r] = m_old * scale + np.log(l_old)
    return out, L


def reference(q, k, v, scale, causal):
    sq, sk = q.shape[0], k.shape[0]
    s = (q.astype(np.float64) @ k.astype(np.float64).T) * scale
    if causal:
        rr, cc = np.meshgrid(np.arange(sq), np.arange(sk), indexing="ij")
        s = np.where(cc - rr > sk - sq, -np.inf, s)
    mx = s.max(1, keepdims=True)
    ok = np.isfinite(mx[:, 0])
    p = np.where(np.isfinite(s), np.exp(s - np.where(np.isfinite(mx), mx, 0)), 0)
    l = p.sum(1)
    o = np.where(ok[:, None], (p @ v.astype(np.float64)) / np.where(l == 0, 1, l)[:, None], 0)
    lse = np.where(ok, mx[:, 0] + np.log(np.where(l == 0, 1, l)), 0)
    return o, lse, ok


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    n = 0
    for d in (32, 64, 96, 128):
        for sq, sk in ((37, 37), (130, 130), (64, 200), (200, 64), (1, 129)):
            for causal in (False, True):
                for bn in (64, 128):
                    for mag in (1.0, 8.0):
                        scale = d ** -0.5
                        q = (rng.standard_normal((sq, d)) * mag).astype(np.float16)
                        k = (rng.standard_normal((sk, d)) * mag).astype(np.float16)
                        v = rng.standard_normal((sk, d)).astype(np.float16)
                        o_a, L_a = run(q, k, v, scale, causal, bn, "orig")
                        o_b, L_b = run(q, k, v, scale, causal, bn, "fast")
                        o_r, L_r, ok = reference(q, k, v, scale, causal)
                        # orig vs fast must agree everywhere (incl. fully masked rows)
                        assert np.allclose(o_a, o_b, rtol=2e-4, atol=2e-5), (d, sq, sk, causal, bn)
                        assert np.allclose(L_a, L_b, rtol=2e-4, atol=2e-4) or \
                            np.array_equal(L_a == -FLT_MAX, L_b == -FLT_MAX), (d, sq, sk, causal, bn)
                        # and match the float64 reference on rows that have a valid key
                        assert np.allclose(o_b[ok], o_r[ok], rtol=5e-4, atol=5e-5), (d, sq, sk, causal, bn)
                        assert np.allclose(L_b[ok], L_r[ok], rtol=5e-4, atol=5e-4), (d, sq, sk, causal, bn)
                        worst = max(worst, float(np.abs(o_b[ok] - o_r[ok]).max()))
                        n += 1
    print(f"OK: {n} configs, orig==fast, both match fp64 reference; worst |dO| = {worst:.2e}")


if __name__ == "__main__":
    main()
