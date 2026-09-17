# Publish source and model assets

[Back to README](../README.md)

Run commands from the repository root.


Extract RNA-GITHUB-SOURCE.zip and upload its contents, including hidden files, to a clean repository. Keep the single README.md at the repository root. The source ZIP includes code, current configurations and regression tests, but excludes training data, weights, generated outputs, development reports, caches and Git history.

Publish RNA-MODEL-ASSETS.zip as a separate GitHub Release asset, not a normal repository file. Confirm source licensing and permission to redistribute the upstream RNA-FM weights before publishing; packaging does not grant a license. No repository or release has been created automatically. Add the actual Release download URL here after publishing; no placeholder download URL is presented as working.

For inference, download the companion asset ZIP from the same Release and extract it into the cloned repository root, preserving its hidden .cache directory. This restores the exact paths used by the commands above. Check hashes in MODEL_ASSETS_MANIFEST.json. Training and asset-dependent tests additionally need the canonical CSV and cluster manifest described above; those data are not included in this source release.

GitHub blocks ordinary Git files larger than 100 MiB. See [GitHub large-file guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Do not commit model binaries, virtual environments, or the delivery ZIP files. The supplied .gitignore keeps them untracked.

From a clean clone with its remote configured:

```bash
git add .
git diff --cached --stat
git commit -m "Add RNA linker source and documentation"
git push
```

Review the staged files before committing. Keep an independent copy of the model assets. No Git LFS setup is needed when distributing weights through Releases.

