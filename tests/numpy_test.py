from ssp.numpy import ssp
import numpy as np
import pytest


N = 32
BATCH_SIZE = 4

""" first, test ssp with reduction """
@pytest.mark.parametrize("shape", [(N,), (N, N), (N, N, N)])
def test_equal_inputs_is_zero(shape):
    x = np.random.random(shape)

    assert ssp(x, x) == 0.0
    assert ssp(np.zeros_like(x), np.zeros_like(x)) == 0.0


@pytest.mark.parametrize("shape", [(N,), (N, N), (N, N, N)])
def test_ssp_against_inverse_is_one(shape):
    x = np.random.random(shape)

    assert ssp(x, -x) == 1.0


@pytest.mark.parametrize("shape", [(N,), (N, N), (N, N, N)])
def test_ssp_against_zero_is_one(shape):
    x = np.random.random(shape)

    assert ssp(x, np.zeros_like(x)) == 1.0



""" now, test with batch dimension """
@pytest.mark.parametrize("shape", [(BATCH_SIZE, N), (BATCH_SIZE, N, N), (BATCH_SIZE, N, N, N)])
def test_equal_inputs_is_zero_batched(shape):
    x = np.random.random(shape)

    result = ssp(x, x, batched=True)

    assert len(result) == BATCH_SIZE
    assert np.allclose(result, 0.0)

    assert np.allclose(ssp(np.zeros_like(x), np.zeros_like(x), batched=True), 0.0)


@pytest.mark.parametrize("shape", [(BATCH_SIZE, N), (BATCH_SIZE, N, N), (BATCH_SIZE, N, N, N)])
def test_ssp_against_inverse_is_one_batched(shape):
    x = np.random.random(shape)

    result = ssp(x, -x, batched=True)

    assert len(result) == BATCH_SIZE
    assert np.allclose(result, 1.0)


@pytest.mark.parametrize("shape", [(BATCH_SIZE, N), (BATCH_SIZE, N, N), (BATCH_SIZE, N, N, N)])
def test_ssp_against_zero_is_one_batched(shape):
    x = np.random.random(shape)

    result = ssp(x, np.zeros_like(x), batched=True)

    assert len(result) == BATCH_SIZE
    assert np.allclose(result, 1.0)
