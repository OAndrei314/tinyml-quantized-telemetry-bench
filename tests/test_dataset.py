import numpy as np
import pytest

from telemetry_bench.dataset import CLASS_NAMES, extract_features, generate_synthetic_telemetry


def test_synthetic_dataset_is_deterministic():
    first = generate_synthetic_telemetry(n_samples=64, seed=42)
    second = generate_synthetic_telemetry(n_samples=64, seed=42)

    np.testing.assert_allclose(first.windows, second.windows)
    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_dataset_shapes_and_classes():
    dataset = generate_synthetic_telemetry(n_samples=80, seq_len=24, n_sensors=3)

    assert dataset.windows.shape == (80, 24, 3)
    assert dataset.features.shape == (80, 3 * 8)
    assert dataset.labels.shape == (80,)
    assert dataset.class_names == CLASS_NAMES
    assert set(dataset.labels.tolist()) == set(range(len(CLASS_NAMES)))


def test_extract_features_rejects_wrong_shape():
    with pytest.raises(ValueError):
        extract_features(np.zeros((12, 4), dtype=np.float32))
