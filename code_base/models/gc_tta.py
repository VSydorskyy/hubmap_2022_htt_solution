from typing import Mapping, Union

import torch
from ttach import SegmentationTTAWrapper
from ttach.base import Merger


class SegmentationTTAWrapperKwargs(SegmentationTTAWrapper):
    def forward(self, image: torch.Tensor, **kwargs) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, **kwargs)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
