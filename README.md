# ML741 Assignment 4

Self-organising maps for network-traffic classification. The report covers
preprocessing, map capacity, codebook clustering and training-label class mapping.
The original 40-cluster assessment is confirmatory. The 150- and 400-cluster
assessments are post-hoc exploratory results.

Python 3.12 and Linux are required. The report also needs LaTeX with IEEEtran,
newtx, latexmk and BibTeX. Figure regeneration needs dvipng. Verification uses
Poppler and MuPDF.

```sh
make setup
make test
make smoke
make report
make verify
```

The PDF and editable LaTeX sources are in `report/`. Saved results and
predictions are in `results/`. No experiment is needed to build the report.

To reproduce the experiments, place the supplied `networkTraffic.csv` and
`features.csv` in `data/`, then run:

```sh
make prepare
make som
make clustering
make assessment
make bootstrap
```

These commands use the fixed partitions, settings and seeds from the report.
They write to `runs/` and leave the saved results unchanged. Training takes
substantial CPU time. Dataset hashes are checked before use. Supplied datasets
are ignored by Git and are not part of the public submission.
