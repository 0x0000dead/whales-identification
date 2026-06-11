# GitHub Wiki Content

This directory contains all 9 pages for the GitHub Wiki.

## Pages

1. **Home.md** - Main landing page with overview and navigation
2. **Installation.md** - Step-by-step installation guide
3. **API-Reference.md** - Complete API documentation with curl examples
4. **Usage.md** - Usage examples for API, Streamlit, notebooks
5. **Architecture.md** - System architecture and technical design
6. **Model-Cards.md** - Detailed model specifications and metrics
7. **Testing.md** - Testing guide and procedures
8. **Contributing.md** - Development workflow and contribution guidelines
9. **FAQ.md** - Frequently asked questions and troubleshooting

## How to Upload to GitHub Wiki

`wiki_content/` is the **source of truth** for the GitHub Wiki. Two sync paths:

### Option 1: Automatic (CI)

The workflow [`.github/workflows/sync-wiki.yml`](../.github/workflows/sync-wiki.yml)
runs on every push to `main` that touches `wiki_content/**` (and on manual
`workflow_dispatch`). It clones the `.wiki.git` repository, copies every page
except this README, and pushes when there is a diff.

> **Do not edit the wiki through the web interface** — changes will be
> overwritten by the next sync. Edit `wiki_content/` and open a PR instead.

### Option 2: Manual (script)

```bash
./scripts/upload_wiki.sh
```

Clones the wiki into a temp directory, copies the pages (excluding this
README), commits and pushes when there is a diff. Reuse an existing clone via
`WIKI_DIR=/path/to/clone ./scripts/upload_wiki.sh`.

## Page Naming Convention

When creating pages on GitHub Wiki, use these exact names (without .md extension):

| File             | GitHub Wiki Page Name |
| ---------------- | --------------------- |
| Home.md          | Home                  |
| Installation.md  | Installation          |
| API-Reference.md | API-Reference         |
| Usage.md         | Usage                 |
| Architecture.md  | Architecture          |
| Model-Cards.md   | Model-Cards           |
| Testing.md       | Testing               |
| Contributing.md  | Contributing          |
| FAQ.md           | FAQ                   |

## Verification

After upload, verify all pages are accessible:

- https://github.com/0x0000dead/whales-identification/wiki
- https://github.com/0x0000dead/whales-identification/wiki/Installation
- https://github.com/0x0000dead/whales-identification/wiki/API-Reference
- ... etc.

## Internal Links

All wiki pages contain cross-links. Ensure links work after upload:

- `[Installation](Installation)` → https://github.com/0x0000dead/whales-identification/wiki/Installation
- `[API Reference](API-Reference)` → https://github.com/0x0000dead/whales-identification/wiki/API-Reference

## Sidebar (Optional)

Create a `_Sidebar.md` file for navigation:

```markdown
## 🐋 Whales Identification

**Quick Start**

- [Home](Home)
- [Installation](Installation)
- [Usage](Usage)

**Documentation**

- [API Reference](API-Reference)
- [Architecture](Architecture)
- [Model Cards](Model-Cards)

**Development**

- [Testing](Testing)
- [Contributing](Contributing)
- [FAQ](FAQ)

**Resources**

- [GitHub Repo](https://github.com/0x0000dead/whales-identification)
- [Hugging Face](https://huggingface.co/baltsat/Whales-Identification)
```

## Footer (Optional)

Create a `_Footer.md` file:

```markdown
---

© 2024 Whales Identification Team | [GitHub](https://github.com/0x0000dead/whales-identification) | [Issues](https://github.com/0x0000dead/whales-identification/issues) | [Discussions](https://github.com/0x0000dead/whales-identification/discussions)
```

## Status

✅ All 9 pages created and ready for upload
✅ Cross-links validated
✅ Content reviewed for accuracy
✅ Code examples tested

## Next Steps

1. Upload pages to GitHub Wiki
2. Test all internal links
3. Add to README.md: Link to Wiki
4. Update GitHub Pages docs/index.md with Wiki links
5. Announce in GitHub Discussions

---

**Created:** September 1, 2025
**Pages:** 9
**Total Content:** ~50,000 words
**Coverage:** Complete project documentation
