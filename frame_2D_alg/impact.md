claude code prompt: run impact.py, trace from its output

plain claude: attach (invariants doc, impact.py output for the field) to the prompt, 

prompt: "Change: <what's changing>. Walk every site in the supplied list,
         check each against the supplied invariants, classify
         no_change | needs_change | needs_decision, then re-scan
         agg_recursion and meta_code for any site the list missed."

impact.py CN.rim

step 0 — resolve the field (the original skipped this):
  parse target class body.
  if field is prop_F_(...)  → expand to backing store:
      rim → {rim, Rt.N_, Rt}                  # view, list store, fork object
      N_ → {N_, Nt.N_, Nt}   H → {H, Nt.H}   L_/B_/C_/X_ likewise
  else (stored base attr: dTT,m,d,c,r,root_…) → flag SHARED across CF/CL/CC/CN,
      site set over-includes, model disambiguates by object type

scope: every FunctionDef in agg_recursion.py + meta_code.py
  (reuse meta_code's ast parse, but NOT its iF_ whitelist —
   writers live in sum2F / Copy_ / add2F, outside it)

outputs:
  reads   : Attribute-load of .rim OR .Rt.N_
  writes  : two syntactically unrelated forms —
      list mutation    : AugAssign / .append on .rim | .Rt.N_     (rim += [L])
      fork replacement : Assign to .Rt                            (N.Rt = sum2F(N.rim))
                         ← contains no `rim` token; name-search is blind to it
  enclosing function per site
  lifecycle touched (resolved, not guessed):
      construct : CN.__init__ (Rt = CF(root=n)), CL(N_=[…]) links entering rims
      copy      : Copy_ Rt branch (fresh fork, shared link elements)
      sum       : sum2F(N.rim) → produces new Rt; add2F if Rt merged
  low-confidence flag: fork bound to a local before access
      (Rt = n.Rt; … Rt.N_ …) — qualifier gone, ownership unrecoverable

NOT an output: obligations. rim symmetry, root_/m_/d_ alignment
  live in the invariants doc, judged by the model on the site set above.

hand to model: site set + changed contract + relevant invariants
  → classify each site: no_change | needs_change | needs_decision