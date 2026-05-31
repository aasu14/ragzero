# Publishing to PyPI

This guide walks through the first release and subsequent updates.

## Pre-flight checklist

Before publishing:

- [ ] **Bump the version** in `src/ragzero/__init__.py` AND `pyproject.toml` (keep them in sync)
- [ ] **Update `CHANGELOG.md`** with the new version's changes
- [ ] **Replace `YOUR_USERNAME`** in `pyproject.toml` and `README.md` with your GitHub username
- [ ] **Verify name availability** at https://pypi.org/project/ragzero/ (404 = available)
- [ ] **All tests pass**: `pytest`
- [ ] **UI builds cleanly**: `cd ui && npm run build`
- [ ] **Commit and tag**: `git tag v0.1.0 && git push --tags`

## Trusted Publishing for this repo (aasu14/ragzero)

This repo is wired for **Trusted Publishing** — GitHub Actions authenticates to PyPI
over OIDC, so **no API token or password is stored anywhere**. The workflow
(`.github/workflows/publish.yml`) already sets `permissions: id-token: write`, the
`pypi` environment, and uses `pypa/gh-action-pypi-publish`; it publishes only on a
`v*` tag pushed to `aasu14/ragzero`.

**One-time PyPI setup** — register a *pending publisher* (works before the project
exists) at <https://pypi.org/manage/account/publishing/> → **Add a new pending
publisher**, entering these values **exactly**:

| Field             | Value         |
|-------------------|---------------|
| PyPI Project Name | `ragzero`     |
| Owner             | `aasu14`      |
| Repository name   | `ragzero`     |
| Workflow name     | `publish.yml` |
| Environment name  | `pypi`        |

> The `Environment name` and `Workflow name` must match the workflow character-for-character,
> or the publish step fails with an OIDC "not authorized" error.
>
> Optional: create a GitHub **Environment** named `pypi` (repo Settings → Environments)
> with a required reviewer, so each publish waits for your approval.

## Releasing (tag → auto-publish)

First release:

```bash
# from the repo root, code committed and tagged v0.1.0
git remote add origin https://github.com/aasu14/ragzero.git
git push -u origin main
git push origin v0.1.0          # this tag push triggers test → build → publish
```

Watch the **Actions** tab; when the `publish` job is green, `pip install "ragzero[all]"`
is live and the page is at <https://pypi.org/project/ragzero/>.

Subsequent releases — never reuse a version number:

```bash
# 1) bump version in pyproject.toml AND src/ragzero/__init__.py (keep in sync)
# 2) add a section to CHANGELOG.md
git add -A && git commit -m "Release v0.2.0"
git tag -a v0.2.0 -m "ragzero 0.2.0"
git push && git push origin v0.2.0
```

(Prefer to do the first upload by hand instead? The manual `twine` flow below also works —
Trusted Publishing and manual uploads are interchangeable once the pending publisher exists.)

## First-time setup

### 1. Create accounts

- **PyPI**: https://pypi.org/account/register/
- **TestPyPI**: https://test.pypi.org/account/register/ (separate account)

Enable 2FA on both.

### 2. Choose your auth method

**Option A — Trusted Publishing (recommended for GitHub Actions)**

1. On PyPI, go to **Publishing → Add a new pending publisher**
2. Fill in:
   - PyPI Project Name: `ragzero`
   - Owner: your GitHub username
   - Repository name: `ragzero`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. The `.github/workflows/publish.yml` workflow in this repo will then publish automatically when you push a `v*` tag.

**Option B — API tokens (for manual publishing)**

```bash
# PyPI: Account → API tokens → Add API token (scope = entire account first time)
# Save the token securely (you only see it once)

# Create ~/.pypirc
cat > ~/.pypirc <<EOF
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
EOF
chmod 600 ~/.pypirc
```

## Building the package

```bash
./build.sh
```

This script:
1. Builds the React UI into `ui/dist/`
2. Calls `python -m build` to produce `dist/ragzero-X.Y.Z-py3-none-any.whl` and `dist/ragzero-X.Y.Z.tar.gz`

The wheel includes the pre-built UI files at `ragzero/ui/dist/` — users don't need npm.

## Validate before uploading

```bash
pip install twine
twine check dist/*       # syntactic check of metadata
```

## Test on TestPyPI first

```bash
twine upload --repository testpypi dist/*
```

Then in a clean virtualenv:

```bash
python -m venv /tmp/test-ragzero
source /tmp/test-ragzero/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            ragzero
ragzero info
ragzero serve --port 8001
```

If `ragzero info` shows the right version, the UI loads at `http://localhost:8001`, and you can ask a question — you're good.

## Publish to real PyPI

```bash
twine upload dist/*
```

Then visit https://pypi.org/project/ragzero/ to confirm.

## Releasing via GitHub Actions (after first publish)

Once Trusted Publishing is set up, you don't run anything locally:

```bash
# Bump version in src/ragzero/__init__.py and pyproject.toml
# Update CHANGELOG.md
git add -A
git commit -m "Release v0.2.0"
git tag v0.2.0
git push && git push --tags
```

The workflow in `.github/workflows/publish.yml` will:
1. Run tests on Python 3.10, 3.11, 3.12
2. Build the UI + wheel + sdist
3. Publish to PyPI using your Trusted Publisher config

## Yanking a bad release

If you push a broken release:

```bash
# On PyPI: Manage → Release → Yank
# Or via CLI:
pip install pypi-cli
pypi yank ragzero 0.1.1 --reason "Critical bug in citation parser"
```

Yanking keeps the file available for pinned installs but hides it from `pip install ragzero`.

You **cannot** delete and re-upload the same version — PyPI rejects this. Bump the version (e.g. 0.1.1 → 0.1.2) and republish.

## Common issues

**"403 Forbidden" on upload**: PyPI doesn't allow file replacement. Bump the version.

**"Invalid token" on upload**: API tokens are version-scoped. If you scoped to a specific project, you can only upload to *that* project.

**Wheel is too big**: PyPI's hard limit is 100MB per file. If `ui/dist` is huge, audit it — typical Vite builds are 2-5MB.

**README doesn't render on PyPI**: PyPI uses GitHub-flavored Markdown. Test rendering at https://github.com (since it uses the same parser).
