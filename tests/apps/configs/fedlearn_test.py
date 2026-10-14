from __future__ import annotations

import sys
from unittest import mock

from taps.apps.configs.fedlearn import FedlearnConfig


def test_create_app() -> None:
    with mock.patch.dict(
        sys.modules,
        {'taps.apps.fedlearn.types': mock.MagicMock()},
    ):
        FedlearnConfig(dataset='MNIST')
        config = FedlearnConfig()

        with mock.patch.dict(
            sys.modules,
            {'taps.apps.fedlearn.app': mock.MagicMock()},
        ):
            config.get_app()


def test_create_app_passes_epochs_and_alpha() -> None:
    app_module = mock.MagicMock()
    with mock.patch.dict(
        sys.modules,
        {
            'taps.apps.fedlearn.types': mock.MagicMock(),
            'taps.apps.fedlearn.app': app_module,
        },
    ):
        FedlearnConfig(epochs=2, batch_size=5, alpha=0.5).get_app()

    kwargs = app_module.FedlearnApp.call_args.kwargs
    assert kwargs['epochs'] == 2  # noqa: PLR2004
    assert kwargs['batch_size'] == 5  # noqa: PLR2004
    assert kwargs['alpha'] == 0.5  # noqa: PLR2004
