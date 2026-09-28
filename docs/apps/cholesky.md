# Tiled Cholesky Decomposition

Computes the Cholesky decomposition of a randomly generated positive definite matrix.
Set `--app.seed` to generate the same matrix on every run.
The lower-triangular factor is saved to `factor.npy` in the run directory.
Based on the tiled algorithm from [this paper](https://www.labri.fr/perso/ejeannot/publications/paap12.pdf){target=_blank}.

Tiled Cholesky decomposition is a canonical example of a dataflow-based workflow because of the inter-task dependencies between different tiles on the matrix.
This application is also data-intensive, with task input/output sizes being $O(b^2)$ where $b$ is the side length of each tile.

## Installation

This application requires numpy which can be installed automatically when installing the TaPS package.
```bash
pip install -e .[cholesky]
```

## Example

The following command computes the decomposition of a 10,000 x 10,000 matrix using 1000 x 1000 block/tile sizes.
```bash
$ python -m taps.run --app cholesky --engine.executor process-pool --app.matrix-size 10000 --app.block-size 1000
```
```bash
[2026-09-28 10:03:41.008] RUN   (taps.run) > Starting benchmark...
...
[2026-09-28 10:03:45.005] APP   (taps.apps.cholesky) > Generated input matrix: (10000, 10000)
[2026-09-28 10:03:45.005] APP   (taps.apps.cholesky) > Block size: 1000
[2026-09-28 10:03:50.224] APP   (taps.apps.cholesky) > Output matrix: (10000, 10000)
[2026-09-28 10:03:50.414] APP   (taps.apps.cholesky) > Saved factor to /private/tmp/taps-example/runs/cholesky_process-pool_2026-09-28-10-03-40/factor.npy
[2026-09-28 10:03:50.415] RUN   (taps.run) > Finished app (elapsed=9.403s)
[2026-09-28 10:03:50.796] RUN   (taps.run) > Finished benchmark (app=cholesky, elapsed=9.799s, tasks=220)
```
Here there are 100 tiles which results in 220 total tasks.
Using a smaller block/tile size would increase the total number of blocks, and therefore the total number of tasks.
This reduces the memory required per-task but also increases the cumulative task overheads.
