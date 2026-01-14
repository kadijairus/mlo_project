import pytest
import torch
from mlo_group_project.model import BreastCancerModel

#Batch size test
@pytest.mark.parametrize("batch_size", [1, 5, 16, 64, 128])
def test_breast_cancer_model_forward_pass_output_shape_matches_batch_size_correctly(batch_size):
    """
    Verifies that the model accepts different batch sizes (1, 5, 16, etc.) 
    and returns an output tensor with shape (batch_size, 1).
    """
    input_features = 30
    model = BreastCancerModel(input_shape=input_features)
    
    # Create input with specific batch size
    dummy_input = torch.randn(batch_size, input_features)
    
    # Push through model
    output = model(dummy_input)
    
    # Assert correct shape
    assert output.shape == (batch_size, 1), \
        f"Output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"

#Deterministic check on eval
def test_breast_cancer_model_returns_identical_outputs_for_same_input_when_in_eval_mode():
    """
    Verifies that when the model is in eval() mode (dropout disabled), 
    running the same input twice produces exactly the same output.
    """
    model = BreastCancerModel(input_shape=30)
    #Send model to eval 
    model.eval() 
    
    input_data = torch.randn(5, 30)
    
    output1 = model(input_data)
    output2 = model(input_data)
    
    assert torch.allclose(output1, output2), \
        "Model output changed between runs despite being in eval mode!"

#Robustness test
@pytest.mark.parametrize("input_tensor", [
    torch.zeros(10, 30),        #All zeros
    torch.ones(10, 30),         #All ones
    torch.randn(10, 30) * 100,  #Huge random numbers
    torch.full((10, 30), -1.0)  #Negative numbers
])
def test_breast_cancer_model_returns_valid_numerical_output_without_nans_or_inf_for_weird_inputs(input_tensor):
    
    model = BreastCancerModel(input_shape=30)
    output = model(input_tensor)
    
    assert not torch.isnan(output).any(), "Model output contained NaNs"
    assert not torch.isinf(output).any(), "Model output contained Infinity"

#Error Handling input
@pytest.mark.parametrize("invalid_shape", [0, -1, -10])
def test_breast_cancer_model_initialization_raises_value_error_when_input_shape_is_non_positive(invalid_shape):
   
    with pytest.raises(ValueError):
        BreastCancerModel(input_shape=invalid_shape)

#Forward integrity check
def test_breast_cancer_model_forward_pass_raises_runtime_error_when_input_tensor_has_wrong_feature_count():
    
    model = BreastCancerModel(input_shape=30)
    wrong_input = torch.randn(10, 5) # Expects 30 features, giving 5
    
    with pytest.raises(RuntimeError):
        model(wrong_input)

#Check Learn, one push
def test_breast_cancer_model_parameters_have_gradients_after_backward_pass_indicating_learnability():
   
    model = BreastCancerModel(input_shape=30)
    x = torch.randn(5, 30)
    y = torch.randn(5, 1) # Dummy targets
    
    output = model(x)
    loss = torch.nn.MSELoss()(output, y)
    loss.backward()
    
    # Check if first layer weights have gradients
    assert model.network[0].weight.grad is not None, "Gradients missing from first layer weights"

#Input feature size adaptability
@pytest.mark.parametrize("input_features", [10, 20, 100, 5])
def test_breast_cancer_model_initialization_works_for_various_input_feature_sizes(input_features):
    """
    Equivalent to your snippet: Ensures the model architecture adapts
    to different input feature counts (columns).
    """
    model = BreastCancerModel(input_shape=input_features)
    dummy_input = torch.randn(1, input_features)
    output = model(dummy_input)

    assert output.shape == (1, 1), f"Expected output (1, 1) for input size {input_features}"
