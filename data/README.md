# Original official datasets

Run `python scripts/download_data.py` to fetch the original upstream IFT/EFT and qa1–qa4 files at the pinned commit. Every file is verified against `manifest.json`. Publication branches keep the manifest instead of duplicating the raw dataset contents. Downloads are ignored by Git but are explicitly verified and included when freezing a training bundle. No data selection or text changes are applied.
