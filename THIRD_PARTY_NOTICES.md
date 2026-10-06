# Third-party notices and asset provenance

This is a public source snapshot. No additional license is granted for original project code, documentation, generated scientific records, trained project checkpoints or project screenshots. The author retains those rights. Third-party material and portions derived from it retain their applicable licenses and notices; this statement neither removes upstream permissions nor relicenses third-party work.

## PushT geometry and simulator conventions

The project uses the public T-body geometry, geometric goal conventions and public actuator behavior of [gym-pusht](https://github.com/huggingface/gym-pusht), version 0.1.6. Copyright 2024 The Hugging Face team. The upstream Apache-2.0 license is copied unchanged to [licenses/gym-pusht-LICENSE](licenses/gym-pusht-LICENSE), and attribution is retained in NOTICE. Apache-2.0 continues to apply to the upstream material and derived portions as applicable; the absence of a project-wide license does not remove that coverage.

Project files `src/pusht_v2.py`, `src/pusht_v4.py`, `src/direct_push.py`, `src/recorded_data.py` and `demo/experiment.js` use or transform those public conventions. The project's geometry/routing helpers, diagnostic controls, observation canonicalization, learning/evaluation logic and recorded viewer are project-specific adaptations rather than an unmodified upstream simulator distribution. The simulator implementation itself is installed as a dependency, not vendored. No upstream endorsement is implied.

## Installed dependencies

Dependency code and wheels are not bundled. Install from the official package sources using the locked requirements. These dependencies retain their own licenses; this source snapshot does not replace them. The installed distribution metadata/licenses were inspected for the versions used:

| Package | Observed license | Use |
| --- | --- | --- |
| gym-pusht 0.1.6 | Apache-2.0 | Original simulator and geometry conventions |
| Gymnasium 1.4.0 | MIT | Environment API |
| Pymunk 6.11.1 | MIT | Native physics used to generate the saved observations |
| PyTorch 2.11.0+cpu | BSD-3-Clause, with bundled-component notices | Project model fitting and checkpoint inference |
| NumPy 2.5.3 | BSD-3-Clause and listed bundled-component licenses | Array operations; consult installed LICENSE.txt |
| Shapely 2.1.2 | BSD-3-Clause; GEOS has separate terms | Public geometry operations |
| Pygame 2.6.1 | LGPL | Upstream rendering dependency; not bundled here |
| Pillow 12.3.0 | MIT-CMU | Image handling dependency |
| Playwright 1.62.0 | Apache-2.0 | Optional browser verification tool; not bundled |

The complete runtime dependency pins are in requirements-pusht-lock.txt. Other transitive packages retain their installed license notices. No dependency license is being relicensed by this project. A future binary/container distribution would need its own dependency-notice review.

## Data, checkpoints and media

The 288 trajectory archives are generated simulator observations/commands from this project's declared TRAIN settings, not a downloaded external human dataset. The three small trained checkpoints were fitted within this project. No external pretrained visual checkpoint is distributed.

The two screenshots show only this project's recorded UI. They were captured from an actual browser and retained byte-for-byte; they are not generated illustrations or photos of unrelated people. The public viewer uses system fonts and local vector/canvas shapes, with no bundled stock photos, web fonts, icons or external media assets. Browser capture code is separate from read-only verification.

Historical pre-execution hash identifiers in result JSON are provenance records, not links to a private service. PUBLIC_PROVENANCE.json records which scientific files were preserved without byte changes. PUBLIC_MANIFEST.json verifies this sanitized distribution independently of the original historical manifests.
