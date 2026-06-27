## Dataset

The designed dataset has 7 features — 5 continuous and 2 categorical.
They are split into two tiers: independent root features (Tier 1)
and features whose values depend on the roots (Tier 2).

### Tier 1 — independent features

#### x1 — bounded continuous index

```
x1 ~ Beta(2, 5)
support: (0, 1)
mean:    0.286
mode:    0.167
```

x1 represents a bounded physiological index — something like a
normalised cardiac ejection fraction. Values are strictly between
0 and 1, skewed toward the lower end. Most patients cluster below
0.4; values above 0.7 are rare.

The hard boundaries are essential: any synthetic record with x1 ≤ 0
or x1 ≥ 1 is a **Type III hallucination** (support violation)
detectable without any population reference.

![x1 distribution](outputs/exploration/x1_distribution.png)

#### x2 — unbounded positive continuous measure

```
x2 ~ Gamma(shape=3, scale=2)
support: (0, ∞)
mean:    6.0
mode:    4.0
```

x2 represents a raw lab measurement — something like serum creatinine
or NT-proBNP. Values are strictly positive with no upper hard boundary,
right-skewed, with most patients in a moderate range and a long tail
toward high values.

The strict positivity is essential: any synthetic record with x2 ≤ 0
is a **Type III hallucination** (support violation) detectable without
any population reference.

x2 is independent of x1 in Tier 1. In Tier 2 both feed into x4 and x5,
where the switch variable x3 determines which of them x5 follows. A
generator that fails to route x1 and x2 correctly through x3 produces
**Type II hallucinations**.

Note: x1 and x2 live on very different scales (mean 0.286 vs 6.0).
Features must be scaled to a common range before computing distances
for privacy-risky detection.

![x2 distribution](outputs/exploration/x2_distribution.png)
