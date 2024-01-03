import abc
from easydict import EasyDict
from numpy import ndarray, zeros
from typing import Dict, List, Optional

from torch import Tensor, device
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure


###############################################################
# Define Metric: PSNR and MSSIM                               #
###############################################################

class Metric(abc.ABC):
    __metaclass__ = abc.ABCMeta
    def __init__(self, name: str) -> None:
        self.name = name
        self.metric = None

    def __str__(self) -> str:
        return f"Metric: {self.name}"
    
    def to(self, device: device) -> None:
        self.metric.to(device)

    @abc.abstractmethod
    def compute(self, y_pred: Tensor, y_true: Tensor) -> ndarray:
        raise NotImplementedError


class PSNR(Metric):
    # see https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html#torchmetrics.image.PeakSignalNoiseRatio
    def __init__(self) -> None:
        super().__init__(name='PSNR')
        self.metric = PeakSignalNoiseRatio()

    def compute(self, y_pred: Tensor, y_true: Tensor) -> ndarray:
        return self.metric(preds=y_pred, target=y_true)
    

class MSSSIM(Metric):
    # see https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html#torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure
    def __init__(self) -> None:
        super().__init__(name='MSSSIM')
        self.metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=7, betas=(0.01, 0.03))

    def compute(self, y_pred: Tensor, y_true: Tensor) -> ndarray:
        return self.metric(preds=y_pred, target=y_true)
    

METRICS: Dict[str, Metric] = {'PSNR': PSNR(), 'MSSSIM': MSSSIM()}


###############################################################
# Define Class Metrics:                                       #
###############################################################

class Metrics():
    def __init__(self, config: EasyDict,
                 device: Optional[device]=device("cpu")
                 ) -> None:
        """ take config.metrics and get all metric s.t. config[metric] == True """
        self.metrics_name: List[str] = list(filter(lambda x: config[x], config))
        self.num_metrics: int = len(self.metrics_name)
        self.metrics: List[Metric] = []

        for metric_name in self.metrics_name:
            if metric_name not in METRICS:
                raise NotImplementedError(f"the metric '{metric_name}' was not implemented. Only {METRICS.keys()} is implemented.")
            self.metrics.append(METRICS[metric_name])
            self.metrics[-1].to(device)
    
    def __str__(self) -> str:
        """ return all the metrics name """
        output = 'Metrics are:\n'
        for i in range(self.num_metrics):
            output += self.metrics_name[i] + '\n'
        return output
    
    def compute(self,
                y_pred: Tensor,
                y_true: Tensor
                ) -> ndarray:
        """ compute all metrics and return a numpy array.
        Tensors must have shape like (B, C, H, W)"""
        metrics_value = zeros(self.num_metrics)
        for i, metric in enumerate(self.metrics):
            metrics_value[i] = metric.compute(y_pred=y_pred, y_true=y_true)
        return metrics_value


if __name__ == '__main__':
    import torch
    shape = (16, 3, 64, 64)
    y = torch.rand(shape)
    x = y + torch.rand(shape) / 10

    device = torch.device("cuda")
    y = y.to(device)
    x = x.to(device)
    
    metrics_config = {'PSNR': True, 'MSSSIM': True}
    metrics = Metrics(config=metrics_config, device=device)
    print(metrics.compute(y_pred=x, y_true=y))
    print(metrics)
    print(metrics.metrics_name)
    
    