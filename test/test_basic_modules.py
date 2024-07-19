import pytest
import nobuco
from nobuco import ChannelOrder
import torch
from torch import nn

def test_extremely_small_tolerance():

    torch_model = nn.Linear(10, 20)
    input = torch.randn((16, 10))

    results = nobuco.pytorch_to_keras(
        torch_model,
        args=[input],
        inputs_channel_order=ChannelOrder.TENSORFLOW,
        trace_shape=True,
        validation_tolerance=1e-20,
    )

    status_value_list = []

    for _, value in results[2].items():

        status_value_list.append(value.status.value)

    assert 3 in status_value_list

def test_linear():

    torch_model = nn.Linear(10, 20)
    input = torch.randn((16, 10))

    results = nobuco.pytorch_to_keras(
        torch_model,
        args=[input],
        inputs_channel_order=ChannelOrder.TENSORFLOW,
        trace_shape=True,
        validation_tolerance=1e-4,
    )

    status_value_list = []

    for _, value in results[2].items():

        status_value_list.append(value.status.value)

    assert 3 not in status_value_list