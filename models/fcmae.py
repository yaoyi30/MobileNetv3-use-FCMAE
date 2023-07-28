# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.nn import init
from timm.models.layers import trunc_normal_
from .mobilenetv3_pretrain import PretrainMobileNetv3Small,PretrainMobileNetv3Large
import torchvision.transforms as T
from PIL import Image
from .mobilenetv3 import Block

class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
                self,
                img_size=224,
                in_chans=3,
                dims=96,
                encoder = None,
                decoder_depth=1,
                decoder_embed_dim=512,
                patch_size=32,
                mask_ratio=0.6,
                norm_pix_loss=False):
        super().__init__()

        # configs
        self.img_size = img_size
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = encoder
        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims,
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(3, decoder_embed_dim, decoder_embed_dim*2, decoder_embed_dim, nn.ReLU, True, 1) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    
    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs,labels=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mobilenet_v3_small(**kwargs):
    model = FCMAE(encoder=PretrainMobileNetv3Small(),dims=576,**kwargs)
    return model

def mobilenet_v3_large(**kwargs):
    model = FCMAE(encoder=PretrainMobileNetv3Large(),dims=960,**kwargs)
    return model


if __name__ == '__main__':
    image = Image.open('ILSVRC2012_val_00045761.JPEG')

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transforms(image).unsqueeze(0)

    model = mobilenet_v3_large()
    model.eval()
    print(model)

    with torch.no_grad():
        out = model(input_tensor)
        print(out)
