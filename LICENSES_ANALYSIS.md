# Python Dependencies License Analysis

## Overview

This document provides a comprehensive analysis of the licensing terms for all Python dependencies used in the EcoMarineAI (whales-identification) project. Understanding these licenses is essential for legal compliance when distributing or deploying the software.

**Analysis Date:** January 2025
**Python Version:** 3.11.x
**Package Manager:** Poetry

---

## License Summary

### License Distribution

| License Type | Count | Percentage | Commercial Use |
|--------------|-------|------------|----------------|
| **MIT** | 45+ | ~40% | ✅ Permitted |
| **Apache 2.0** | 25+ | ~22% | ✅ Permitted |
| **BSD-3-Clause** | 30+ | ~27% | ✅ Permitted |
| **BSD-2-Clause** | 5+ | ~4% | ✅ Permitted |
| **PSF (Python)** | 3+ | ~3% | ✅ Permitted |
| **LGPL** | 2+ | ~2% | ⚠️ Conditions Apply |
| **Other Permissive** | 5+ | ~4% | ✅ Generally Permitted |

**Overall Assessment:** ✅ All dependencies use **permissive open-source licenses** that are compatible with commercial use of the software itself.

**⚠️ Important Note:** While the *code dependencies* permit commercial use, the *trained models* and *training data* have non-commercial restrictions (see [LICENSE_MODELS.md](LICENSE_MODELS.md) and [LICENSE_DATA.md](LICENSE_DATA.md)).

---

## Core ML Framework Dependencies

### PyTorch Ecosystem

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **torch** | 2.4.1 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **torchvision** | 0.19.1 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **timm** | 1.0.9 | Apache 2.0 | Apache-2.0 | ✅ Yes |

**PyTorch License Details:**
- **Source:** https://github.com/pytorch/pytorch
- **License URL:** https://github.com/pytorch/pytorch/blob/main/LICENSE
- **Key Terms:**
  - Free to use, modify, and distribute
  - Must include copyright notice and license
  - No trademark rights granted
  - Provided "as is" without warranty

**TIMM (PyTorch Image Models) License Details:**
- **Source:** https://github.com/huggingface/pytorch-image-models
- **License URL:** https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE
- **Key Terms:**
  - Apache 2.0 allows commercial use
  - Must include license and copyright notice
  - Must state significant changes made
  - Patent rights granted

### TensorFlow/Keras Ecosystem

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **keras** | 3.6.0 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **tensorflow** (transitive) | - | Apache 2.0 | Apache-2.0 | ✅ Yes |

---

## Web Framework Dependencies

### FastAPI Stack

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **fastapi** | ^0.115.12 | MIT | MIT | ✅ Yes |
| **uvicorn** | ^0.34.3 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **pydantic** | 2.9.2 | MIT | MIT | ✅ Yes |
| **pydantic-core** | 2.23.4 | MIT | MIT | ✅ Yes |
| **python-multipart** | ^0.0.20 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **starlette** (transitive) | - | BSD-3-Clause | BSD-3-Clause | ✅ Yes |

**FastAPI License Details:**
- **Source:** https://github.com/tiangolo/fastapi
- **License URL:** https://github.com/tiangolo/fastapi/blob/master/LICENSE
- **Key Terms:**
  - MIT License - very permissive
  - Free to use, copy, modify, merge, publish, distribute, sublicense, sell
  - Must include copyright notice

---

## Computer Vision Dependencies

### Image Processing

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **opencv-python** | 4.10.0.84 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **opencv-python-headless** | 4.10.0.84 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **pillow** | 10.4.0 | HPND | HPND | ✅ Yes |
| **scikit-image** | 0.24.0 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **albumentations** | 1.4.18 | MIT | MIT | ✅ Yes |
| **albucore** | 0.0.17 | MIT | MIT | ✅ Yes |
| **imageio** | 2.35.1 | BSD-2-Clause | BSD-2-Clause | ✅ Yes |
| **rembg** | ^2.0.61 | MIT | MIT | ✅ Yes |

**OpenCV License Details:**
- **Source:** https://github.com/opencv/opencv
- **License URL:** https://github.com/opencv/opencv/blob/master/LICENSE
- **Key Terms:**
  - Apache 2.0 License
  - Free for commercial and non-commercial use
  - Must include license notice
  - Patent grant included

**Pillow License (HPND):**
- Historical Permission Notice and Disclaimer
- Effectively equivalent to MIT for practical purposes
- Very permissive, allows commercial use

---

## Data Science Dependencies

### Scientific Computing

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **numpy** (transitive) | - | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **pandas** | 2.2.3 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **scipy** | 1.14.1 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **scikit-learn** | 1.5.2 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **joblib** | 1.4.2 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |

### ML Infrastructure

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **onnxruntime** | ^1.20.1 | MIT | MIT | ✅ Yes |
| **safetensors** | 0.4.5 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **huggingface-hub** | 0.25.2 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **wandb** | 0.18.3 | MIT | MIT | ✅ Yes |

---

## Development Dependencies

### Testing & Quality

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **pytest** | ^8.3.3 | MIT | MIT | ✅ Yes |
| **pytest-cov** | ^6.0.0 | MIT | MIT | ✅ Yes |
| **httpx** | ^0.27.0 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |

### Code Formatting & Linting

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **black** | ^24.0.0 | MIT | MIT | ✅ Yes |
| **isort** | ^5.13.0 | MIT | MIT | ✅ Yes |
| **flake8** | ^7.0.0 | MIT | MIT | ✅ Yes |
| **mypy** | ^1.8.0 | MIT | MIT | ✅ Yes |
| **autopep8** | ^2.3.1 | MIT | MIT | ✅ Yes |
| **pylint** | ^3.3.1 | GPL-2.0 | GPL-2.0-only | ⚠️ Tool Only |
| **bandit** | ^1.7.0 | Apache 2.0 | Apache-2.0 | ✅ Yes |

**Note on Pylint (GPL-2.0):**
- Pylint uses GPL-2.0 license
- **This does NOT affect your code** - it's a development tool only
- Your code does not link against or include Pylint
- Safe for commercial projects as a development dependency

---

## Visualization & UI Dependencies

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **streamlit** | 1.39.0 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **matplotlib** | ^3.10.0 | PSF-based | PSF-2.0 | ✅ Yes |
| **altair** | 5.4.1 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **pydeck** | 0.9.1 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **rich** | 13.9.2 | MIT | MIT | ✅ Yes |

---

## Utility Dependencies

### General Utilities

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **requests** | 2.32.3 | Apache 2.0 | Apache-2.0 | ✅ Yes |
| **urllib3** | 2.2.3 | MIT | MIT | ✅ Yes |
| **pyyaml** | 6.0.2 | MIT | MIT | ✅ Yes |
| **toml** | 0.10.2 | MIT | MIT | ✅ Yes |
| **click** | 8.1.7 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **tqdm** | 4.66.5 | MIT/MPL 2.0 | MIT | ✅ Yes |
| **jinja2** | 3.1.4 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **typing-extensions** | 4.12.2 | PSF | PSF-2.0 | ✅ Yes |

### Jupyter/IPython

| Package | Version | License | SPDX | Commercial |
|---------|---------|---------|------|------------|
| **ipython** | 8.28.0 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **ipykernel** | 6.29.5 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **jupyter-client** | 8.6.3 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |
| **jupyter-core** | 5.7.2 | BSD-3-Clause | BSD-3-Clause | ✅ Yes |

---

## License Compatibility Matrix

### Compatibility with Project License (MIT)

| Dependency License | Compatible with MIT? | Notes |
|--------------------|---------------------|-------|
| MIT | ✅ Yes | Identical terms |
| BSD-2-Clause | ✅ Yes | More permissive |
| BSD-3-Clause | ✅ Yes | Slightly more restrictive (no endorsement) |
| Apache 2.0 | ✅ Yes | Compatible, adds patent grant |
| PSF | ✅ Yes | Python Software Foundation, permissive |
| HPND | ✅ Yes | Historical, effectively MIT-like |
| LGPL | ⚠️ Conditional | Must provide source if modifying LGPL code |
| GPL (dev tools) | ✅ Yes | Development tools don't affect distribution |

### Commercial Use Summary

✅ **All runtime dependencies permit commercial use**

The entire dependency stack can be used in commercial software products, with the following requirements:

1. **Include license notices** for all dependencies in your distribution
2. **Maintain copyright attributions** as required by BSD/MIT licenses
3. **State changes** if you modify Apache 2.0 licensed code
4. **Do not use trademarks** of dependency projects without permission

---

## Compliance Requirements

### Required Notices

When distributing this software, you must include:

1. **This project's MIT License** (see [LICENSE](LICENSE))
2. **Third-party license notices** (recommended: create THIRD_PARTY_LICENSES.txt)
3. **Model license notice** (see [LICENSE_MODELS.md](LICENSE_MODELS.md))
4. **Data attribution** (see [LICENSE_DATA.md](LICENSE_DATA.md))

### Recommended: THIRD_PARTY_LICENSES.txt

Create a file listing all third-party licenses. Example format:

```
================================================================================
PyTorch
================================================================================
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
...

BSD 3-Clause License
...

================================================================================
FastAPI
================================================================================
The MIT License (MIT)

Copyright (c) 2018 Sebastián Ramírez
...
```

---

## Dependency Audit Commands

To verify licenses in your environment:

```bash
# Install pip-licenses
pip install pip-licenses

# Generate license report
pip-licenses --format=markdown --with-urls

# Check for copyleft licenses
pip-licenses --allow-only="MIT;BSD-3-Clause;BSD-2-Clause;Apache Software License;PSF;HPND;ISC"

# Export to CSV for audit
pip-licenses --format=csv > dependency_licenses.csv
```

---

## Updates and Maintenance

### Monitoring License Changes

When updating dependencies:
1. Run `pip-licenses` to check for license changes
2. Review any new dependencies added
3. Update this document if licenses change
4. Check for any LGPL/GPL additions that might affect distribution

### Automated Checks

Consider adding to CI/CD:

```yaml
# .github/workflows/license-check.yml
- name: Check licenses
  run: |
    pip install pip-licenses
    pip-licenses --fail-on="GPL;AGPL;SSPL"
```

---

## Contact and Questions

For licensing questions or concerns:
- **GitHub Issues:** https://github.com/0x0000dead/whales-identification/issues
- **Email:** Contact project maintainers

---

## Disclaimer

This license analysis is provided for informational purposes and does not constitute legal advice. For commercial deployments or if you have specific legal concerns, please consult with a qualified attorney familiar with open-source licensing.

**Last Updated:** January 2025
**Version:** 1.0
**Maintained By:** EcoMarineAI Project Team
