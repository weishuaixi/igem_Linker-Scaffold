# Model identity and provenance

[Back to README](../README.md)

Commands below run from the repository root.

## Final model identity

Both runs produced the filename `rna-linker-v5-34-1.3714.ckpt`, but their bytes differ. A filename or rounded validation loss is not a unique model identifier.

| Run | Checkpoint SHA-256 | Evidence status |
| --- | --- | --- |
| Original V5 | `91bb5d3be35428c0ede186235bec12d836eb4c235622dbd3d95a917193ed3b50` | Historical expanded generation and conditioning results above |
| September 10 replacement | `2a6bab7bfb4afc145c2309c38798e2db43f085d628c291ab4004b33932546ced` | Training completed; exported file integrity verified; detailed benchmark must be rerun |

The replacement checkpoint is 533,679,315 bytes. Its local backup archive, `RNA-BEST-DOWNLOAD-20260910-042140-147668.tar.gz`, has SHA-256 `255ec9de3319a3d92ea090f98d5ef351859c2355cf9c810e7dd8352dc2131dab`. The archive includes the best checkpoint, training manifest, configuration, installed-environment record and package checksums. Runtime weights and backup archives remain outside normal Git tracking; publish runtime assets separately if their licenses permit redistribution.

RNA-FM is 1,194,424,423 bytes; its SHA-256 is `5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99`. `MODEL_ASSETS_MANIFEST.json` records weight hashes. `SOURCE_MANIFEST.json` records source hashes. The companion SHA256SUMS file verifies both ZIP archives.

