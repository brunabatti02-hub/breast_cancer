import torch
import torch.nn as nn
import timm


class BackboneWrapper(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
        )
        self.out_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)


class HFTNet(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, freeze_backbones=False):
        super().__init__()

        self.densenet = BackboneWrapper("densenet201", pretrained=pretrained)
        self.xception = BackboneWrapper("xception", pretrained=pretrained)
        self.vit = BackboneWrapper("vit_base_patch16_224", pretrained=pretrained)
        self.deit = BackboneWrapper("deit_base_patch16_224", pretrained=pretrained)
        self.swin = BackboneWrapper("swin_tiny_patch4_window7_224", pretrained=pretrained)

        if freeze_backbones:
            for m in [self.densenet, self.xception, self.vit, self.deit, self.swin]:
                for p in m.parameters():
                    p.requires_grad = False

        self.proj_dense = nn.Linear(self.densenet.out_dim, 768)
        self.proj_xcep = nn.Linear(self.xception.out_dim, 768)
        self.proj_vit = nn.Linear(self.vit.out_dim, 768)
        self.proj_deit = nn.Linear(self.deit.out_dim, 768)
        self.proj_swin = nn.Linear(self.swin.out_dim, 768)

        self.fusion_proj = nn.Linear(5 * 768, 768)
        self.norm1 = nn.LayerNorm(768)

        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=16, batch_first=True)
        self.norm2 = nn.LayerNorm(768)

        self.ffn = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(3072, 768),
        )
        self.norm3 = nn.LayerNorm(768)

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        fd = self.densenet(x)
        fx = self.xception(x)
        fv = self.vit(x)
        fde = self.deit(x)
        fs = self.swin(x)

        fd = self.proj_dense(fd)
        fx = self.proj_xcep(fx)
        fv = self.proj_vit(fv)
        fde = self.proj_deit(fde)
        fs = self.proj_swin(fs)

        feats = torch.stack([fv, fs, fde, fd, fx], dim=1)
        concat_feats = feats.reshape(feats.size(0), -1)

        fused = self.fusion_proj(concat_feats)
        fused = self.norm1(fused)

        fused_seq = fused.unsqueeze(1)
        attn_out, _ = self.mha(fused_seq, fused_seq, fused_seq)
        x1 = self.norm2(fused_seq + attn_out)

        ffn_out = self.ffn(x1)
        x2 = self.norm3(x1 + ffn_out)

        out = x2.squeeze(1)
        logits = self.classifier(out)
        return logits
