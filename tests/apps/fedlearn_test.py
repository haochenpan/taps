from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any
from unittest import mock

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from taps.apps.fedlearn.app import FedlearnApp  # noqa: E402
from taps.apps.fedlearn.app import write_results  # noqa: E402
from taps.apps.fedlearn.types import DataChoices  # noqa: E402
from taps.engine import Engine  # noqa: E402

SAMPLES = 300


def _fake_data(
    data_name: DataChoices,
    root: pathlib.Path,
    train: bool,
    download: bool = False,
) -> Any:
    # A small MNIST-shaped dataset, so no test downloads MNIST.
    generator = torch.Generator().manual_seed(1 if train else 2)
    size = SAMPLES if train else SAMPLES // 4
    inputs = torch.rand((size, 1, 28, 28), generator=generator)
    targets = torch.randint(0, 10, (size,), generator=generator)
    return torch.utils.data.TensorDataset(inputs, targets)


def _run(
    engine: Engine,
    run_dir: pathlib.Path,
    **options: Any,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        'clients': 2,
        'rounds': 2,
        'dataset': DataChoices.MNIST,
        'batch_size': 1,
        'epochs': 1,
        'lr': 1e-3,
        'data_dir': run_dir / 'data',
        'seed': 0,
    }
    kwargs.update(options)
    run_dir.mkdir()
    with mock.patch('taps.apps.fedlearn.app.load_data', _fake_data):
        app = FedlearnApp(**kwargs)
    app.run(engine, run_dir)
    app.close()
    with open(run_dir / 'results.json') as f:
        return json.load(f)


def _without_times(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in results:
        for history in result['histories']:
            for record in history:
                record.pop('time')
        if result['test'] is not None:
            result['test'].pop('time')
    return results


def test_fedlearn_app_writes_each_rounds_results(
    engine: Engine,
    tmp_path: pathlib.Path,
) -> None:
    results = _run(engine, tmp_path / 'run', epochs=2)

    assert [result['round_idx'] for result in results] == [0, 1]
    for result in results:
        assert sorted(result['clients']) == [0, 1]
        for client_idx, history in zip(
            result['clients'],
            result['histories'],
            strict=True,
        ):
            # Each client trains on about half of the samples, one per
            # batch, and logs its mean loss every 100 batches.
            assert [record['epoch'] for record in history] == [0, 1]
            for record in history:
                assert record['client_idx'] == client_idx
                assert record['round_idx'] == result['round_idx']
                assert record['batch_idx'] == 99  # noqa: PLR2004
                assert record['train_loss'] > 0
                datetime.datetime.fromisoformat(record['time'])
        test = result['test']
        assert test['round_idx'] == result['round_idx']
        assert test['test_loss'] > 0
        datetime.datetime.fromisoformat(test['time'])


def test_fedlearn_app_without_testing_writes_no_test_result(
    engine: Engine,
    tmp_path: pathlib.Path,
) -> None:
    results = _run(engine, tmp_path / 'run', rounds=1, test=False)

    assert [result['test'] for result in results] == [None]


def test_fedlearn_app_with_a_seed_writes_the_same_results(
    engine: Engine,
    tmp_path: pathlib.Path,
) -> None:
    first = _without_times(_run(engine, tmp_path / 'first'))
    second = _without_times(_run(engine, tmp_path / 'second'))
    other = _without_times(_run(engine, tmp_path / 'other', seed=1))

    assert first == second
    assert first != other


def test_write_results_writes_a_datetime_in_iso_format(
    tmp_path: pathlib.Path,
) -> None:
    logged = datetime.datetime(2026, 1, 2, 3, 4, 5, 6)
    write_results([{'time': logged}], tmp_path / 'results.json')

    with open(tmp_path / 'results.json') as f:
        assert json.load(f) == [{'time': '2026-01-02T03:04:05.000006'}]

    with pytest.raises(TypeError, match='set is not JSON serializable'):
        write_results([{'clients': {0}}], tmp_path / 'other.json')
