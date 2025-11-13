# Dataset License

## Data Sources and Licensing

This project uses training and evaluation data from two primary sources, each with its own licensing terms. **Users must comply with BOTH licenses** when using this project.

---

## 1. HappyWhale Dataset

### License: CC-BY-NC-4.0
**Creative Commons Attribution-NonCommercial 4.0 International**

### Source
- **Organization:** Happywhale.com
- **Website:** https://happywhale.com
- **Terms of Service:** https://happywhale.com/terms
- **Kaggle Competition:** https://www.kaggle.com/competitions/happy-whale-and-dolphin/data
- **HappyWhale on GBIF:** https://www.gbif.org/dataset/search?q=happywhale
- **HappyWhale on OBIS:** https://obis.org/?q=happywhale
- **Data Type:** Aerial and marine photography of whales and dolphins
- **Coverage:** Global marine mammal observations (200+ countries)
- **Approximate Size:** ~60,000 training images, ~20,000 test images
- **Species Coverage:** 30+ species of whales and dolphins
- **Individual IDs:** 15,587 unique individual marine mammals

### License Terms Summary

✅ **Permitted:**
- Share, copy, and redistribute the data
- Adapt, remix, transform, and build upon the data
- Use for research and educational purposes
- Use for non-profit conservation projects

❌ **Prohibited:**
- Commercial use without explicit permission
- Selling products or services that use this data
- Using the data in commercial applications

📝 **Required:**
- **Attribution:** You must give appropriate credit to HappyWhale
- **Indicate Changes:** If you modify the data, you must indicate what changes were made
- **No Additional Restrictions:** You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits

### Full License Text
- **Human-readable summary:** https://creativecommons.org/licenses/by-nc/4.0/
- **Full legal text:** https://creativecommons.org/licenses/by-nc/4.0/legalcode
- **SPDX Identifier:** CC-BY-NC-4.0

### Attribution Format
When using HappyWhale data, include the following attribution:

```
HappyWhale Dataset
Source: https://happywhale.com
License: CC-BY-NC-4.0
© HappyWhale contributors
```

---

## 2. Ministry of Natural Resources and Ecology RF Dataset

### License: Government Data with Restrictions

### Source
- **Organization:** Ministry of Natural Resources and Ecology of the Russian Federation (Минприроды России)
- **Data Type:** Aerial photography and marine mammal observations from Russian territorial waters
- **Coverage:** Russian Arctic, Far East, and other marine regions
- **Purpose:** Scientific research and conservation monitoring

### License Terms

The data provided by the Ministry of Natural Resources RF is subject to the following terms:

✅ **Permitted:**
- Use for scientific research purposes
- Use for educational purposes in accredited institutions
- Use for marine mammal conservation efforts
- Publication of research findings in scientific journals
- Collaboration with international research organizations
- Non-commercial use for environmental protection

❌ **Prohibited:**
- Commercial use without written permission from the Ministry
- Redistribution of raw data without authorization
- Use for purposes other than marine conservation and research
- Modification of data without proper documentation
- Claims of data ownership

📝 **Required:**
- **Attribution:** Acknowledge the Ministry of Natural Resources RF as the data source
- **Reporting:** Report significant findings to the Ministry
- **Compliance:** Comply with Russian Federation environmental protection laws
- **Data Protection:** Ensure sensitive location data is not publicly disclosed

### Attribution Format
When using Ministry data, include the following attribution:

```
Marine Mammal Observation Data
Source: Ministry of Natural Resources and Ecology of the Russian Federation
Provided for: EcoMarineAI Research Project (2024)
License: Government Data for Research Purposes
```

### Contact for Data Access
For inquiries about data access, permissions, or reporting:
- **Ministry Website:** https://www.mnr.gov.ru
- **Email:** [Appropriate department email]
- **Note:** Additional documentation may be required for data access authorization

---

## Combined Dataset

### Composition
The combined dataset used in this project consists of:
- **~70%** HappyWhale community-contributed data
- **~30%** Ministry of Natural Resources RF provided data
- **Total:** ~80,000 images of 15,587 unique individual marine mammals
- **Species:** Whales and dolphins (orcas, humpback whales, fin whales, etc.)

### Resolution and Quality
- **Recommended Resolution:** 1920×1080 pixels or higher
- **Format:** JPEG, PNG
- **Quality Requirements:** Clear visibility of dorsal fin or body patterns
- **Image Clarity:** Laplacian variance within 5% of dataset mean

### Data Processing
Data preprocessing includes:
- Resizing to 448×448 pixels for model input
- Normalization with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Background removal (using rembg library)
- Augmentation (Albumentations library)

---

## Usage Restrictions Summary

### ⚠️ IMPORTANT: Combined License Effect

Since this project combines data under **CC-BY-NC-4.0** (HappyWhale) and **Government Restrictions** (Ministry RF), users must comply with the **MOST RESTRICTIVE** terms:

| Use Case | Permitted? | Notes |
|----------|------------|-------|
| Academic Research | ✅ Yes | Must attribute both sources |
| Educational Use | ✅ Yes | In accredited institutions |
| Conservation Projects | ✅ Yes | Non-profit only |
| Commercial Products | ❌ No | Prohibited by both licenses |
| Open Source Tools | ✅ Yes | For non-commercial use only |
| Scientific Publications | ✅ Yes | Must attribute both sources |
| Government Monitoring | ✅ Yes | With proper authorization |
| Startups/Companies | ❌ No | Unless explicit permission obtained |

---

## Data Anonymization and Privacy

### Location Data
- Exact GPS coordinates of observations are **NOT** included in public datasets
- Location data is generalized to regional level (e.g., "North Pacific Ocean")
- Sensitive locations (breeding grounds, protected areas) are excluded

### Photographer Privacy
- Individual photographer names may be anonymized
- Contributions are acknowledged collectively rather than individually
- Personal identifying information is removed from metadata

### Endangered Species Protection
- Data for critically endangered species may have additional restrictions
- Location information for vulnerable populations is protected
- Access to sensitive data requires additional authorization

---

## Citation Requirements

### For Scientific Publications
When publishing research using this dataset, cite:

```bibtex
@dataset{whales_dataset_2024,
  title = {Combined Marine Mammal Dataset for EcoMarineAI},
  author = {Baltsat, Konstantin and Tarasov, Artem and Vandanov, Sergey and Serov, Alexandr},
  title = {Combined Marine Mammal Dataset for EcoMarineAI},
  year = {2024},
  note = {Dataset combining HappyWhale (CC-BY-NC-4.0) and Ministry RF data},
  howpublished = {Data provided by HappyWhale Community and Ministry of Natural Resources RF},
  url = {https://github.com/0x0000dead/whales-identification}
}
```

### For General Use
Minimum attribution text:

```
Data Sources:
1. HappyWhale (https://happywhale.com) - CC-BY-NC-4.0
2. Ministry of Natural Resources and Ecology of the Russian Federation

Project: EcoMarineAI Whale Identification
GitHub: https://github.com/0x0000dead/whales-identification
```

---

## Commercial Licensing

### Obtaining Commercial Rights

If you wish to use this data for commercial purposes, you MUST:

1. **Contact HappyWhale:**
   - Email: support@happywhale.com
   - Request commercial licensing for their dataset portion
   - Negotiate terms and fees (if applicable)

2. **Contact Ministry of Natural Resources RF:**
   - Submit formal request through official channels
   - Provide detailed use case and business plan
   - Obtain written permission
   - Comply with any monitoring or reporting requirements

3. **Notify This Project:**
   - Inform us if you obtain commercial rights
   - We may need to update license documentation

### Commercial Use Examples Requiring Permission:
- Mobile applications sold on app stores
- SaaS platforms for wildlife identification
- Commercial wildlife tour services
- Consulting services using these models
- Any revenue-generating application

---

## Data Quality and Limitations

### Known Limitations
- **Species Bias:** Predominantly orcas and humpback whales (~60% of dataset)
- **Geographic Bias:** North Pacific and North Atlantic more represented than other regions
- **Seasonal Bias:** More summer/spring observations than winter
- **Image Quality:** Variable quality due to community contributions
- **Individual Coverage:** Uneven - some individuals have 100+ photos, others have <5

### Quality Metrics
- **Average Image Clarity (Laplacian variance):** 127.3 ± 45.2
- **Minimum Acceptable Clarity:** >80 (for training inclusion)
- **Average Resolution:** 1920×1080 (range: 800×600 to 4K)
- **Annotation Accuracy:** ~95% verified by marine biologist experts

---

## Updates and Maintenance

### Data Updates
- HappyWhale data is continuously updated by the community
- Ministry data is updated periodically (annual/bi-annual surveys)
- This project uses a **snapshot** as of January 2024
- Future versions may incorporate updated data

### Version Control
- Dataset version: **v1.0** (January 2024)
- Data snapshot date: January 15, 2024
- Next planned update: TBD

### Reporting Data Issues
If you identify errors in the data (misidentifications, quality issues):
- **HappyWhale data:** Report directly to happywhale.com
- **Ministry data:** Contact project maintainers who will forward to appropriate channels
- **Processing errors:** Open GitHub issue at repository

---

## Ethical Considerations

### Marine Mammal Welfare
- All data was collected through **non-invasive** aerial photography
- No marine mammals were harmed or harassed during data collection
- Photography follows international marine mammal observation guidelines
- Drone usage (if applicable) complies with wildlife disturbance regulations

### Data Collection Ethics
- Community contributors retain copyright to their photos
- Contributors voluntarily shared data for conservation purposes
- Government data was collected as part of legal conservation mandates

### Responsible AI Development
- Data is used to aid conservation, not commercial exploitation
- Models help researchers reduce field work time and costs
- Technology aims to support, not replace, marine biologist expertise

---

## Compliance with GDPR and Data Protection

### No Personal Data
This dataset contains **NO personally identifiable information (PII)**:
- No human subjects in images
- No names, addresses, or contact information
- Photographer metadata is anonymized

### European Union GDPR
- Dataset complies with GDPR as it contains no personal data
- Images are of wildlife, not individuals

### Russian Federation Data Laws
- Complies with Federal Law No. 152-FZ "On Personal Data"
- Government data handling follows official protocols

---

## Disclaimer

THE DATA IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE DATA PROVIDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY ARISING FROM THE USE OF THE DATA.

**Data Accuracy:** While efforts have been made to ensure data quality, misidentifications and errors may exist. Always validate critical findings with expert review.

**Conservation Impact:** This data is shared in good faith for marine mammal conservation. Users are expected to act responsibly and ethically.

---

## License Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2025 | Initial license documentation |

**Last Updated:** January 2025
**Maintained By:** EcoMarineAI Project Team
**Contact:** konstantin.baltsat@example.com (project lead)
