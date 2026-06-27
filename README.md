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
