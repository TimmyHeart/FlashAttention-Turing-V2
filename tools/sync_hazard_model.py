"""
Randomised hazard model of the FLASH_TURING_SEPV synchronisation scheme (CPU only).

Each of 8 "warps" executes the op sequence of the forward KV loop; ops are scheduled in
random interleavings. A shared-memory buffer is modelled as 8 slices (one per writing
warp) each carrying a version tag. A read of buffer X for iteration i asserts that
every slice carries version i (RAW) - an early overwrite by a faster warp (WAR) shows up
as a version mismatch. __syncthreads is a real barrier across all warps.

Sequences modelled (per KV iteration i):
  V1 (aliased K/V in one buffer KV):  W(KV,K_i) BAR R(KV,K_i) BAR W(KV,V_i) BAR R(KV,V_i) BAR
  SEPV:                               W(K,K_i)  BAR R(K_i)    W(V,V_i) BAR R(V,V_i)
then the epilogue: W(Q-buffer as O) BAR R(Q-buffer as O).
Q is read every iteration with version 'Q'; the epilogue overwrites it with 'O'.

Sanity: the model must also CATCH a deliberately broken variant (SEPV with the V-store
barrier removed), otherwise it proves nothing.
"""
import random

W = 8


def program(mode, iters, drop_bar=None):
    ops = [("wQ",), ("BAR",)] if False else []
    prog = []
    # Q was written before the loop by all warps (version 'Q'), then:
    for i in range(iters):
        if mode == "v1":
            prog += [("W", "KV", ("K", i)), ("BAR",), ("R", "KV", ("K", i)), ("R", "Q", "Q"), ("BAR",),
                     ("W", "KV", ("V", i)), ("BAR",), ("R", "KV", ("V", i)), ("BAR",)]
        else:
            prog += [("W", "K", i), ("BAR",), ("R", "K", i), ("R", "Q", "Q"), ("W", "V", i)]
            if drop_bar != "vstore":
                prog += [("BAR",)]
            prog += [("R", "V", i)]
    prog += [("W", "Q", "O"), ("BAR",), ("R", "Q", "O")]
    return prog


def simulate(mode, iters, seed, drop_bar=None):
    rng = random.Random(seed)
    prog = program(mode, iters, drop_bar)
    mem = {"KV": [None] * W, "K": [None] * W, "V": [None] * W, "Q": ["Q"] * W}
    pc = [0] * W
    at_bar = [False] * W
    while any(pc[w] < len(prog) for w in range(W)):
        runnable = [w for w in range(W) if pc[w] < len(prog) and not at_bar[w]]
        if not runnable:
            # everybody left is at a barrier -> release
            for w in range(W):
                if at_bar[w]:
                    at_bar[w] = False
                    pc[w] += 1
            continue
        w = rng.choice(runnable)
        op = prog[pc[w]]
        if op[0] == "BAR":
            at_bar[w] = True
            if all(at_bar[x] or pc[x] >= len(prog) for x in range(W)) and all(at_bar):
                for x in range(W):
                    at_bar[x] = False
                    pc[x] += 1
            continue
        if op[0] == "W":
            mem[op[1]][w] = op[2]
        else:
            for sl in range(W):
                if mem[op[1]][sl] != op[2]:
                    return False
        pc[w] += 1
    return True


def main():
    for mode in ("v1", "sepv"):
        ok = all(simulate(mode, 5, s) for s in range(3000))
        print(f"{mode:5s}: 3000 random schedules, 5 KV iterations -> {'no hazard' if ok else 'HAZARD'}")
        assert ok
    bad = [s for s in range(3000) if not simulate("sepv", 5, s, drop_bar="vstore")]
    print(f"negative control (SEPV without the V-store barrier): hazard caught in {len(bad)}/3000 schedules")
    assert len(bad) > 0


if __name__ == "__main__":
    main()
