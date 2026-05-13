# Bioconda submission notes

This directory contains the PrismSNV Bioconda recipe draft.

Before opening a Bioconda pull request:

1. Commit the package files in this repository.
2. Create and push a release tag that matches `pyproject.toml`, for example:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

3. Download the source archive and compute its SHA256:

   ```bash
   curl -L -o prismsnv-0.1.0.tar.gz \
     https://github.com/xjtu-omics/PrismSNV/archive/refs/tags/v0.1.0.tar.gz
   sha256sum prismsnv-0.1.0.tar.gz
   ```

4. Replace these placeholders in `meta.yaml`:

   - `TODO_REPLACE_WITH_SHA256_AFTER_TAGGING_V0.1.0`
   - `TODO_ADD_GITHUB_USERNAME`

5. Copy `meta.yaml` into a fork of `bioconda-recipes`:

   ```text
   bioconda-recipes/recipes/prismsnv/meta.yaml
   ```

6. Run the Bioconda lint/build checks from the `bioconda-recipes` checkout,
   then open a pull request against `bioconda/bioconda-recipes`.
