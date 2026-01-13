import pytest

@pytest.mark.parametrize("input_size, expected_output", [
    (10, 2),
    (20, 2),
    (100, 2),
    (5, 2),
    (0, 2)  # Edge case
])
def test_model_shape(input_size, expected_output):
    model = MyModel(input_size=input_size)
    assert model.forward(data).shape == (1, expected_output)
