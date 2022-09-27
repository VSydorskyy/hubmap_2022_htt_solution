import torch
from segmentation_models_pytorch import FPN, Unet, UnetPlusPlus
from segmentation_models_pytorch.base import initialization as init


class FPNChLast(FPN):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        self.encoder = self.encoder.to(memory_format=torch.channels_last)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)
        for i in range(len(features)):
            features[i] = features[i].to(memory_format=torch.contiguous_format)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class UnetChLast(Unet):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        self.encoder = self.encoder.to(memory_format=torch.channels_last)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)
        for i in range(len(features)):
            features[i] = features[i].to(memory_format=torch.contiguous_format)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class UnetPlusPlusChLast(UnetPlusPlus):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        self.encoder = self.encoder.to(memory_format=torch.channels_last)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)
        for i in range(len(features)):
            features[i] = features[i].to(memory_format=torch.contiguous_format)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
