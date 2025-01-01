import torch
import pytest
from torch import cuda
from trifast.compile_helpers import ParamLookup, DtypeParamLookup


@pytest.fixture
def mock_dtype_lookup(monkeypatch):
    def mock_from_file(path):
        return None

    monkeypatch.setattr(DtypeParamLookup, "from_file", mock_from_file)


@pytest.fixture
def mock_cuda_capability(monkeypatch):
    def mock_get_device_capability():
        return (8, 6)

    monkeypatch.setattr(cuda, "get_device_capability", mock_get_device_capability)


@pytest.fixture
def config_dir(tmp_path):
    configs = [(7, 5), (8, 0), (8, 6), (9, 0)]
    for dtype in [torch.bfloat16, torch.float32, torch.float16]:
        for major, minor in configs:
            (tmp_path / f"{major}_{minor}" / str(dtype)).mkdir(parents=True)
            (tmp_path / f"{major}_{minor}" / str(dtype) / "fwd.yaml").touch()
    return tmp_path


def test_exact_match(mock_cuda_capability, mock_dtype_lookup, config_dir):
    result = ParamLookup.from_folder(config_dir, "fwd.yaml")
    assert str(result.root) == str(config_dir / "8_6")


def test_major_match_lower_minor(
    mock_cuda_capability, mock_dtype_lookup, monkeypatch, config_dir
):
    def mock_cap():
        return (8, 9)

    monkeypatch.setattr(cuda, "get_device_capability", mock_cap)
    result = ParamLookup.from_folder(config_dir, "fwd.yaml")
    assert str(result.root) == str(config_dir / "8_6")


def test_lower_major(mock_cuda_capability, mock_dtype_lookup, monkeypatch, config_dir):
    def mock_cap():
        return (9, 5)

    monkeypatch.setattr(cuda, "get_device_capability", mock_cap)
    result = ParamLookup.from_folder(config_dir, "fwd.yaml")
    assert str(result.root) == str(
        config_dir / "9_0"
    )  # Changed from config_path to root


def test_no_compatible_version(
    mock_cuda_capability, mock_dtype_lookup, monkeypatch, config_dir
):
    def mock_cap():
        return (7, 0)

    monkeypatch.setattr(cuda, "get_device_capability", mock_cap)
    with pytest.raises(ValueError, match="No compatible configuration found"):
        ParamLookup.from_folder(config_dir, "fwd.yaml")
