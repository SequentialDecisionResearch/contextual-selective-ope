# Pre-publication assembly audit

- Added missing common modules and remaining author-provided source scripts from `codes(1).zip`.
- Corrected `run_all.py` import `run_obd` → `run_obdnew` (function still named `run_obd`).
- Added `baseline_is_behavior=False` option to OBD core and, if true, enforced w0=1 and dr0=reward. This is consistent with the author-provided final OBD manifest; **only mock tests performed**.
- Made config use a portable default repository directory and environment variables; changed default mode to `smoke` to prevent accidental long runs. **This is not the original historical execution environment.**
- Added three untouched original run manifests.
- Added seven small archived final summary tables, preserving original root `results/` artifacts rather than silently overwriting old versions.
- Rewrote README and added reproducibility / manuscript correction notes.
- No full experiment, raw data download, source LaTeX manuscript edit, or PDF recompile performed.
