from keras import ops, layers, Sequential, random
import pytest
from itertools import product

from ssp import SSP1D, SSP2D, fftfreq


NX: int = 512
DX: float = 4000 / NX
DT: float = 0.1

OMEGA: ops.array = fftfreq(n=NX, d=DT)
K: ops.array = fftfreq(n=NX, d=DX)


LOSSES: dict = {
    1: {"fn": SSP1D, "rank": 1},
    2: {"fn": SSP2D, "rank": 2},
}

LOSSES_1D: dict = {idx: d for idx, d in enumerate(LOSSES.values()) if d["rank"] == 1}


@pytest.mark.parametrize(["loss_idx", "lowpass"], list(product(LOSSES, [None, "static", "adaptive"])))
def test_serialization(loss_idx, lowpass):
    fn = LOSSES[loss_idx]["fn"]
    rank = LOSSES[loss_idx]["rank"]

    if lowpass:
        if rank == 1:
            # test 1D losses with time series
            f = OMEGA
        else:
            # test 2D losses with spatial data
            f = K
    else:
        f = []

    loss_fn = fn(lowpass=lowpass, f=f)
    cfg = loss_fn.get_config()

    # copy from config
    loss_fn_from_config = fn.from_config(cfg)
    cfg_from_config = loss_fn_from_config.get_config()

    for k, v in cfg.items():
        if not k == "name":
            assert v == cfg_from_config[k], f"{v} does not match!"


@pytest.mark.parametrize("loss_idx", LOSSES)
def test_lowpass_raise_value_error(loss_idx):
    with pytest.raises(ValueError):
        LOSSES[loss_idx]["fn"](lowpass="None")


@pytest.mark.parametrize(["loss_idx", "lowpass"], list(product(LOSSES, [None, "static", "adaptive"])))
def test_training(loss_idx, lowpass):
    fn = LOSSES[loss_idx]["fn"]
    rank = LOSSES[loss_idx]["rank"]

    if lowpass:
        if rank == 1:
            # test 1D losses with time series
            f = OMEGA
        else:
            # test 2D losses with spatial data
            f = K
            return
    else:
        f = []

    loss_fn = fn(lowpass=lowpass, f=f, f_filter=2.5)

    model = Sequential([layers.Conv1D(filters=1, kernel_size=3, padding="same"), layers.Reshape(target_shape=(512,))])

    x = ops.ones((5, 512, 1))
    y = ops.ones((5, 512))

    model.build(input_shape=x.shape)
    model.compile(
        optimizer="adam",
        loss=loss_fn
    )

    model.fit(x=x, y=y, batch_size=1)


@pytest.mark.parametrize(["loss_fn", "lowpass"], list(product([(SSP1D, 1), (SSP2D, 2)], [None, "static", "adaptive"])))
def test_ssp_range(loss_fn, lowpass):
    loss, rank = loss_fn
    if lowpass:
        if rank == 1:
            # test 1D losses with time series
            f = OMEGA
        else:
            # test 2D losses with spatial data
            f = K
    else:
        f = []

    loss = loss(lowpass=lowpass, f=f, f_filter=6.0)

    x1 = ops.ones((1, *[NX]*rank))
    x2 = ops.ones((1, *[NX]*rank)) + random.normal((1, *[NX]*rank))

    if lowpass:
        assert loss(x1, x2) > 0.0
        assert loss(x1, x2) <= 1.0
        assert loss(x1, x1) >= 0.0
        assert loss(x2, x2) >= 0.0

    else:
        assert loss(x1, x1) == 0.0
        assert loss(x2, x2) == 0.0
        assert loss(x1, x2) > 0.0
        assert loss(x1, x2) < 1.0
        assert loss(x1, -x1) == 1.0
        