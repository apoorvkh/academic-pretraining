import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from ..models import BaseModelClass


def count_flops_per_example(model_class: BaseModelClass) -> float:
    assert torch.cuda.is_available()

    model = model_class.build_model(use_custom_kernels=False)
    model = nn.Module.cuda(model)

    dataset = model_class.load_dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    datapoint = next(iter(dataloader))
    for k in datapoint:
        datapoint[k] = datapoint[k].cuda()

    flop_counter = FlopCounterMode(display=True)

    with flop_counter:
        # number of flops are expected to be constant regardless of precision
        # using mixed precision to save memory here
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model(**datapoint).get("loss").backward()

    flops_per_example = flop_counter.get_total_flops()
    return flops_per_example
