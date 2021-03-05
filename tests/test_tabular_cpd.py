import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from graphical_models.src import tabular_wrapper as tw


@pytest.fixture
def wrapper():
    return tw.TabCPD(inferencer="naive")


@pytest.fixture
def data_numeric():
    return pd.DataFrame(
        {
            "target": [0., 1, 0, 0, 1, 1, 0, 0, 0, 1],
            "field_1": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "field_2": [2, 3, 3, 2, 2, 2, 2, 3, 3, 3]
        }
    )

#
# def test_init(wrapper):
#     assert wrapper.target == "target"


def test_fit(wrapper, data_numeric):
    expected_index = pd.MultiIndex.from_product([
        [0, 1],
        [2, 3],
        [0.0, 1.0]
    ], names=["field_1", "field_2", "target"])
    expected = pd.DataFrame(
        {
            "pr_e": [0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2],
            "pr_joint": [0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1],
            "pr_cond": [0.5, 0.5, 0.666, 0.333, 0.666, 0.333, 0.5, 0.5]
        },
        index=expected_index
    )
    actual = wrapper.fit(data_numeric.loc[:, ["field_1", "field_2"]], data_numeric.loc[:, "target"])
    assert_frame_equal(actual, expected, rtol=10e-3)

