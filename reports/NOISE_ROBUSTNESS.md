# Noise robustness benchmark

Measures accept rate of the CLIP anti-fraud gate when positives are
corrupted by three realistic noise sources. The ТЗ (Параметр 4) demands
that accuracy drop by ≤ 20% under such conditions.

| Variant | Accepted / Total | Accept rate | Mean score | Drop vs clean | ≤ 20% target |
|---|---|---|---|---|:---:|
| `clean` | 29/30 | 0.9667 | 0.9580 | +0.0 % | ✓ |
| `gaussian_sigma25` | 27/30 | 0.9000 | 0.8655 | +6.9 % | ✓ |
| `jpeg_q20` | 29/30 | 0.9667 | 0.9300 | +0.0 % | ✓ |
| `blur_r4` | 27/30 | 0.9000 | 0.8832 | +6.9 % | ✓ |

## Variant recipes
- `clean`: untouched RGB image
- `gaussian_sigma25`: per-pixel N(0, 25²) noise
- `jpeg_q20`: re-encoded as JPEG quality 20
- `blur_r4`: PIL Gaussian blur radius 4
