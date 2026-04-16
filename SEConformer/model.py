from torch import nn
import torch


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 16)
        self.fc2 = nn.Linear(channels // 16, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SEResConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.se = SEBlock(out_c)

        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)
        out += residual
        return torch.relu(out)


class ConformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out

        x2 = self.norm2(x)
        x = x + self.ff(x2)

        return x


class SEConformer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.b1 = SEResConvBlock(3, 32)
        self.b2 = SEResConvBlock(32, 64)
        self.b3 = SEResConvBlock(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(128 * 8 * 8, 256)

        self.conformer = ConformerBlock(256)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        x = x.unsqueeze(1)
        x = self.conformer(x)
        x = x.mean(dim=1)

        return self.classifier(x)
