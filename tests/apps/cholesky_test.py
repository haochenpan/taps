from __future__ import annotations

import pathlib

import numpy
import pytest

from taps.apps.cholesky import CholeskyApp
from taps.apps.cholesky import create_psd_matrix
from taps.engine import Engine


@pytest.mark.parametrize(('matrix_size', 'block_size'), ((4, 4), (16, 4)))
def test_cholesky_app(
    matrix_size: int,
    block_size: int,
    engine: Engine,
    tmp_path: pathlib.Path,
) -> None:
    app = CholeskyApp(matrix_size, block_size)
    app.run(engine, tmp_path)
    app.close()

    assert [path.name for path in tmp_path.iterdir()] == ['factor.npy']
    factor = numpy.load(tmp_path / 'factor.npy')
    assert factor.shape == (matrix_size, matrix_size)
    assert factor.dtype == numpy.float64


def test_mismatched_size_error() -> None:
    with pytest.raises(
        ValueError,
        match='The matrix size must be greater than or equal to the block',
    ):
        CholeskyApp(16, 5)


def test_create_psd_matrix_seed() -> None:
    first = create_psd_matrix(8, seed=0)
    second = create_psd_matrix(8, seed=0)
    other = create_psd_matrix(8, seed=1)
    assert numpy.array_equal(first, second)
    assert not numpy.array_equal(first, other)


@pytest.mark.parametrize(
    ('matrix_size', 'block_size'),
    ((4, 4), (8, 4), (16, 4), (24, 6)),
)
def test_cholesky_app_factor(
    matrix_size: int,
    block_size: int,
    engine: Engine,
    tmp_path: pathlib.Path,
) -> None:
    app = CholeskyApp(matrix_size, block_size, seed=0)
    app.run(engine, tmp_path)
    app.close()

    factor = numpy.load(tmp_path / 'factor.npy')
    matrix = create_psd_matrix(matrix_size, seed=0)
    numpy.testing.assert_allclose(factor, numpy.linalg.cholesky(matrix))
