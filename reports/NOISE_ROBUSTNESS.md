# Noise robustness benchmark

Measures accept rate of the CLIP anti-fraud gate when positives are
corrupted by three realistic noise sources. The ТЗ (Параметр 4) demands
that accuracy drop by ≤ 20% under such conditions.

| Variant | Accepted / Total | Accept rate | Mean score | Drop vs clean | ≤ 20% target |
|---|---|---|---|---|:---:|
| `clean` | 95/100 | 0.9500 | 0.9445 | +0.0 % | ✓ |
| `gaussian_sigma25` | 95/100 | 0.9500 | 0.9178 | +0.0 % | ✓ |
| `jpeg_q20` | 96/100 | 0.9600 | 0.9425 | -1.1 % | ✓ |
| `blur_r4` | 96/100 | 0.9600 | 0.9500 | -1.1 % | ✓ |

## Variant recipes
- `clean`: untouched RGB image
- `gaussian_sigma25`: per-pixel N(0, 25²) noise
- `jpeg_q20`: re-encoded as JPEG quality 20
- `blur_r4`: PIL Gaussian blur radius 4
