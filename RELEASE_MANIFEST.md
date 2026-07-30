# Release manifest

## Included

- Minimal OpenGait runtime needed by the HDA-Gait V4 pretraining entry point.
- Multi-domain dataset parsing, balanced sampling, and independent q/k views.
- ResNet9 domain backbone with input-adaptive IN/SyncBN.
- SupCon, MMD, GRL, and adversarial domain-classification code.
- Sanitized six-V100 configuration.
- Reproducible Python environment files.
- Import/configuration smoke tests.

## Deliberately excluded

- `output/`, checkpoints, TensorBoard events, and training logs.
- All PKL datasets and the private 207,915-identity partition.
- Small private dataset samples and compressed archives.
- Migration bundles and packed Conda environments.
- Paper figures, synthetic schematic plots, and analysis caches.
- Upload scripts and cloud credentials.
- Fine-tuning experiments and unrelated OpenGait model implementations.
- Machine-specific absolute paths.

## Before making the GitHub repository public

1. Confirm the upstream OpenGait redistribution terms.
2. Choose and add a compatible `LICENSE`.
3. Add author, contact, paper, and citation metadata.
4. Run `python tests/check_release.py`.
5. Review `git status` before the first commit.
6. Keep datasets and checkpoints on external artifact hosting.

