from telemetry_bench.dataset import generate_synthetic_telemetry
from telemetry_bench.metrics import accuracy_score
from telemetry_bench.models import QuantizedLinearClassifier, train_linear_classifier


def test_fp32_and_int8_models_classify_synthetic_telemetry():
    train = generate_synthetic_telemetry(n_samples=256, seed=10)
    test = generate_synthetic_telemetry(n_samples=128, seed=11)

    fp32 = train_linear_classifier(train)
    int8 = QuantizedLinearClassifier.from_fp32(fp32, train.features)

    fp32_accuracy = accuracy_score(test.labels, fp32.predict(test.features))
    int8_accuracy = accuracy_score(test.labels, int8.predict(test.features))

    assert fp32_accuracy >= 0.90
    assert int8_accuracy >= 0.88
    assert fp32_accuracy - int8_accuracy <= 0.08


def test_int8_classifier_is_smaller_than_fp32():
    train = generate_synthetic_telemetry(n_samples=128, seed=3)

    fp32 = train_linear_classifier(train)
    int8 = QuantizedLinearClassifier.from_fp32(fp32, train.features)

    assert int8.q_weights.dtype.kind == "i"
    assert int8.q_weights.itemsize == 1
    assert int8.classifier_nbytes() < fp32.classifier_nbytes()
    assert int8.nbytes() < fp32.nbytes()
