# Note: due to the external binary and data dependencies of the montage_wrapper
# package, these tests are *heavily* mocked. They essentially just test that
# the code is executable.
from __future__ import annotations

import contextlib
import pathlib
import sys
from typing import Generator
from unittest import mock

import pytest

from taps.apps.montage import bgexec_prep
from taps.apps.montage import configure_montage
from taps.apps.montage import madd
from taps.apps.montage import mbackground
from taps.apps.montage import mdiff
from taps.apps.montage import mimgtbl
from taps.apps.montage import MontageApp
from taps.apps.montage import moverlaps
from taps.apps.montage import mproject
from taps.engine import Engine


@pytest.fixture(autouse=True, scope='module')
def _mock_montage() -> Generator[None, None, None]:
    functions = [
        'mImgtbl',
        'mMakeHdr',
        'mProject',
        'mOverlaps',
        'mDiff',
        'mFitExec',
        'mBgModel',
        'mBackground',
    ]

    mods = {'montage_wrapper': mock.MagicMock()}
    with mock.patch.dict(sys.modules, mods), contextlib.ExitStack() as stack:
        for function in functions:
            stack.enter_context(mock.patch(f'montage_wrapper.{function}'))

        yield


def test_configure_montage_spec(tmp_path: pathlib.Path) -> None:
    configure_montage(
        img_folder=tmp_path / 'img_folder',
        img_tbl=tmp_path / 'img_tbl',
        img_hdr=tmp_path / 'img_hdr',
    )


def test_mproject_spec(tmp_path: pathlib.Path) -> None:
    mproject(
        input_path=tmp_path / 'input_path',
        template_path=tmp_path / 'template_path',
        output_path=tmp_path / 'output_path',
    )


def test_mimgtbl_spec(tmp_path: pathlib.Path) -> None:
    mimgtbl(img_dir=tmp_path / 'img_dir', tbl_path=tmp_path / 'tbl_path')


def test_moverlaps_spec(tmp_path: pathlib.Path) -> None:
    moverlaps(img_tbl=tmp_path / 'img_tbl', diffs_tbl=tmp_path / 'diffs_tbl')


def test_mdiff_spec(tmp_path: pathlib.Path) -> None:
    mdiff(
        image_1=tmp_path / 'image_1',
        image_2=tmp_path / 'image_2',
        template=tmp_path / 'template',
        output_path=tmp_path / 'output_path',
    )


def test_bgexec_prep_spec(tmp_path: pathlib.Path) -> None:
    bgexec_prep(
        img_table=tmp_path / 'img_table',
        diffs_table=tmp_path / 'diffs_table',
        diff_dir=tmp_path / 'diff_dir',
        output_dir=tmp_path / 'output_dir',
    )


def test_mbackground_spec(tmp_path: pathlib.Path) -> None:
    mbackground(
        in_image=tmp_path / 'in_image',
        out_image=tmp_path / 'out_image',
        a=1,
        b=1,
        c=1,
    )


def test_madd_spec(tmp_path: pathlib.Path) -> None:
    madd(
        images_table=tmp_path / 'images_table',
        template_header=tmp_path / 'template_header',
        out_image=tmp_path / 'out_image',
        corr_dir=tmp_path / 'corr_dir',
    )


def test_montage_app(engine: Engine, tmp_path: pathlib.Path) -> None:
    img_folder = tmp_path / 'images'
    img_folder.mkdir()
    for i in range(3):
        (img_folder / f'{i}.fits').touch()

    app = MontageApp(img_folder)

    diffs_tbl = tmp_path / 'diffs_tbl.csv'
    with open(diffs_tbl, 'w') as f:
        f.write("""\
| cntr1 | cntr2 | plus | minus | diff |
| int   | int   | char |  char | char |
 0 1 image1.fits image2.fits diff.fits
""")

    corrections_tbl = tmp_path / 'corrections.csv'
    with open(corrections_tbl, 'w') as f:
        f.write("""\
| id | a | b | c |
   0   0   0   0
   1   0   0   0
   4   0   0   0
""")

    img_tbl = tmp_path / 'images.csv'
    with open(img_tbl, 'w') as f:
        f.write("""\
fitshdr
image.fits
""")

    with (
        mock.patch(
            'taps.apps.montage.moverlaps',
            autospec=True,
            return_value=diffs_tbl,
        ),
        mock.patch(
            'taps.apps.montage.bgexec_prep',
            autospec=True,
            return_value=corrections_tbl,
        ),
        mock.patch(
            'taps.apps.montage.mimgtbl',
            autospec=True,
            return_value=img_tbl,
        ),
    ):
        app.run(engine, tmp_path)

    app.close()


_DIFFS_TABLE = """\
| cntr1 | cntr2 | plus | minus | diff |
| int   | int   | char |  char | char |
 0 1 image1.fits image2.fits diff.fits
"""
_CORRECTIONS_TABLE = """\
| id | a | b | c |
   0   0   0   0
   1   0   0   0
   4   0   0   0
"""
_IMAGES_TABLE = """\
fitshdr
image.fits
"""


@contextlib.contextmanager
def _montage_tools() -> Generator[dict[str, mock.MagicMock], None, None]:
    """Mock the Montage tools the app calls; each writes the table it would."""

    def write(path: str, text: str) -> None:
        pathlib.Path(path).write_text(text)

    def imgtbl(img_dir: str, tbl_path: str) -> None:
        write(tbl_path, _IMAGES_TABLE)

    def overlaps(img_tbl: str, diffs_tbl: str) -> None:
        write(diffs_tbl, _DIFFS_TABLE)

    def bgmodel(img_table: str, fits_tbl: str, corrections_tbl: str) -> None:
        write(corrections_tbl, _CORRECTIONS_TABLE)

    side_effects = {
        'mImgtbl': imgtbl,
        'mOverlaps': overlaps,
        'mBgModel': bgmodel,
    }
    with contextlib.ExitStack() as stack:
        tools = {
            name: stack.enter_context(
                mock.patch(
                    f'montage_wrapper.{name}',
                    side_effect=side_effects.get(name),
                ),
            )
            for name in (
                'mImgtbl',
                'mMakeHdr',
                'mProject',
                'mOverlaps',
                'mDiff',
                'mFitExec',
                'mBgModel',
                'mBackground',
                'mAdd',
            )
        }
        yield tools


def _image_folder(tmp_path: pathlib.Path) -> pathlib.Path:
    img_folder = tmp_path / 'images'
    img_folder.mkdir()
    for i in range(3):
        (img_folder / f'{i}.fits').touch()
    return img_folder


def test_montage_app_coadds_corrected_images(
    engine: Engine,
    tmp_path: pathlib.Path,
) -> None:
    run_dir = tmp_path / 'run'
    app = MontageApp(_image_folder(tmp_path))

    with _montage_tools() as tools:
        app.run(engine, run_dir)
    app.close()

    output_dir = run_dir / 'data'
    corrected_tbl = output_dir / 'corrected_images.tbl'
    assert (
        mock.call(str(output_dir / 'corrections'), str(corrected_tbl))
        in tools['mImgtbl'].call_args_list
    )
    tools['mAdd'].assert_called_once_with(
        str(corrected_tbl),
        str(output_dir / 'Kimages.hdr'),
        str(output_dir / 'm17.fits'),
        str(output_dir / 'corrections'),
    )
