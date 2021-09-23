import torch

from nat_base import NAGenerator


class ToyGenerator(NAGenerator):

    @torch.no_grad()
    def generate(self, models, sample, **unused):
        finalized = super().generate(models, sample, **sample)
