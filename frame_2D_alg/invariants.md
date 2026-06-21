# agg_recursion / meta_code — structural invariants

The contracts the aliased-mutation couplings depend on, extracted from the class
definitions and the sum/merge/copy primitives. A change to one of these objects is
safe iff it preserves these. Each is anchored to the code that establishes it.
Where your own comments flag a point as unsettled, it's marked **(unsettled)**.

## 1. Storage lives in the forks, not on the node

CN's `N_, C_, L_, B_, X_, rim, H` are **property views** onto fork objects, not stored
attributes (`prop_F_`, class body of `CN`):

- `n.rim` ≡ `n.Rt.N_`
- `n.N_` ≡ `n.Nt.N_`  ·  `n.L_` ≡ `n.Lt.N_`  ·  `n.B_` ≡ `n.Bt.N_`  ·  `n.C_` ≡ `n.Ct.N_`  ·  `n.X_` ≡ `n.Xt.N_`
- `n.H` ≡ `n.Nt.H`

**Invariant:** the six forks `Nt, Lt, Bt, Ct, Xt, Rt` are the actual storage; the CN
attributes are aliases. A write to `n.rim += [L]` mutates `n.Rt.N_`; `n.Nt.N_ = …` and
`n.N_ = …` are the same store. Any analysis of `.rim` must also catch `.Rt.N_`, and vice-versa.

## 2. Aggregation: `c` is the weight, `dTT` is primary

`sum2F` and `add2F` are the only summation primitives:

- `F.c = Σ n.c` — **additive**.
- `F.dTT = Σ n.dTT·(n.c/F.c)` — **c-weighted mean**. Same form for `r, kern, span, yx`.
- `F.box = extend_box(…)` — **union** over members (typ==3), not a mean.
- `add2F` maintains the same means incrementally: `F.dTT = F.dTT·_w + n.dTT·w`, `_w,w = F.c/C, n.c/C`.

**Invariant:** `c` is additive and is the weight for every other aggregate; `dTT/r/kern/span/yx`
are c-weighted means; `box` is a union. Any change to what `c` means, or to how dTT
aggregates, must change `sum2F`, `add2F`, **and** `sum_vt` (which repeats the c-weighting)
together. `__bool__ = bool(c)`, so a zero-c object is falsy/empty and `if N:` tests `c`.

## 3. `m, d` are cached projections of `dTT`

`vt_(TT)` defines them: `m = (m_/(m_+|d_|)) @ wTT[0]`, symmetric for `d`. `Fvt_` writes them back.

**Invariant:** `m, d` are never independent state — always `vt_(dTT, wTT)`. Every mutation of
`dTT` must be followed by `Fvt_`/`vt_`, or `m/d` desync. `dTT` is `(2,9)`; the 9 params are fixed
(`wM, wD, wi, wG, wI, wa, wL, wS, wA`).

## 4. Link ↔ rim symmetry

`comp_N` builds `L = CL(N_=[_N,N])`, then for typ==3 nodes:
`_N.rim += [L]; N.rim += [L]; n.compared.add(_n)`.

**Invariant:** for any link `L` with `L.N_=[a,b]`: `L ∈ a.rim` **and** `L ∈ b.rim`, and
`a, b ∈ L.N_` (rim is symmetric, stored in `Rt.N_`). `_n ∈ n.compared` ⟺ they were linked.
Holds only for `typ==3`; lower-typ nodes (PPs) don't accumulate rim.

## 5. Centroid membership: `root_ / m_ / d_` stay index-aligned

`sum2F` with `m_,d_` (centroid formation) appends per member in one step:
`N.m_ += [m]; N.d_ += [d]; N.root_ += [F]`.

**Invariant:** for every node `N`, `root_[i], m_[i], d_[i]` describe membership in the *same*
centroid, fixed at append time. The double buffer `_root_/_m_/_d_` holds the prior iteration,
equally aligned (`k = n._root_.index(_C); n._m_[k], n._d_[k]`). Promotion is a **triple** move
(`n._root_=n.root_; n._m_=n.m_; n._d_=n.d_`); `cluster_C` resets all six together.

**Merge** (`comp_C_`): `n.root_[n.root_.index(C)] = _C` replaces in place to "keep m,d positions" —
merging two centroids must preserve each member's index alignment, never reorder `root_`
independently of `m_/d_`.

**(unsettled)** whether `root_` ever holds B-roots as well as centroids (`# Cs not Bs?`), and
whether a promoted centroid's own `m_/d_` collides with its per-member `m_/d_`
(`# may conflict with promoted C m_,d_?`). Treat both as fragile, not law.

## 6. Fork ownership

`comb_Ft`: `G = CN(Nt,Lt,Bt,Ct); Nt.root = Lt.root = Bt.root = Ct.root = G`.
`add_Nt`: `N.root = G` for members.

**Invariant:** a G's four forks point back to G via `.root`; G's member nodes carry `.root = G`.
`root` (singular) is the one connectivity parent; `root_` (plural) is the set of centroids —
distinct fields, distinct meaning, do not conflate.

## 7. `Copy_` is shallow at the leaves, deep in the forks

`Copy_`: `N_=copy(N.N_)` (shallow — new list, **shared elements**); forks `Nt..Rt` recursively
`Copy_`'d (fresh objects); `angl` deep-copied; `root_=list(N.root_)` (new list, shared centroids).

**Invariant:** a copied CN shares its member nodes/links with the original but owns independent
fork objects. Mutating a member *through* the copy is visible in the original; mutating a fork is
not. Any change that relies on copy-isolation must check which side of this line it sits on.

## 8. `typ` ↔ class

`sum2F` promotes via `cls_[typ]`, `cls_=[CF, CL, CC, CN]` (typ 0/1/2/3). rim updates gate on
`typ==3`; comp depth gates on `typ`. Class chain is `CF → CL → CC` and `CF → CL → CN`
(CN extends CL, **not** CC, though both carry `m_/d_`).

**Invariant:** `typ` selects the class on summation and gates behavior; a node's `typ` must match
its actual class and role (0 fork, 1 link, 2 centroid, 3 node).

## 9. Cross-module param-count boundary

meta_code's CN uses `dTT (2,9)`; frame_blobs' CN uses `derTT (2,8)`; comp_slice uses `verT (2,6)`.

**Invariant:** the comparand-vector width changes across the pipeline boundary. Anything crossing
frame_blobs / comp_slice → agg_recursion must map `(2,6)`/`(2,8)` → `(2,9)`; a change to the
9-param layout is a boundary change, not a local one.