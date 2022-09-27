import random

from albumentations import HistogramMatching
from albumentations.augmentations.domain_adaptation import apply_histogram


class HistogramMatchingPerClass(HistogramMatching):
    def apply(self, img, reference_image=None, blend_ratio=0.5, **params):
        # Pick image with respect to class
        reference_image = self.read_fn(random.choice(self.reference_images[params["class_name"]]))
        return apply_histogram(img, reference_image, blend_ratio)

    def get_params(self):
        return {
            "reference_image": None,
            "blend_ratio": random.uniform(self.blend_ratio[0], self.blend_ratio[1]),
        }

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)
        params.update({"class_name": kwargs["class_name"]})
        return params
