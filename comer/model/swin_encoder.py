from typing import Tuple

import pytorch_lightning as pl
import torch
from  timm.models.swin_transformer import SwinTransformer
from torch import FloatTensor, LongTensor
from einops import rearrange


class SwinEncoder(pl.LightningModule):
    def __init__(self, d_model: int):
        super().__init__()
        self.swin = SwinTransformer(
            img_size=224,
            in_chans=1,
            patch_size=4,
            window_size=7,

            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],

            mlp_ratio=4,
            # global_pool='',
            num_classes=0,
        )
        self.linear = torch.nn.Linear(
            in_features=self.swin.num_features,
            out_features=d_model,
        )
    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:

        img_feature = self.swin.forward_features(img)
        img_feature = self.linear(img_feature)
        print(img_feature.shape)
        img_feature = rearrange(img_feature, 'b (h w) c -> b h w c', h=7, w=7)

        out_mask = img_mask[:, 0::4, 0::4] # After Patch cutting
        out_mask = out_mask[:, 0::2, 0::2] # After Block 1
        out_mask = out_mask[:, 0::2, 0::2] # After Block 2
        out_mask = out_mask[:, 0::2, 0::2] # After Block 3

        return img_feature, out_mask

if __name__ == "__main__":
    # check output shape
    encoder = SwinEncoder(d_model=512)
    img = torch.randn(2, 1, 224, 224)
    img_mask = torch.ones(2, 224, 224).long()  # assuming img_mask is of shape [b, h', w']
    out, out_mask = encoder(img, img_mask)
    print("Output shapes:")
    print(out.shape)  # should be [2, 512]
    print(out_mask.shape)  # should be [2, 14, 14] if img was 224x224