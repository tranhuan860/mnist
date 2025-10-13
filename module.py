import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, depth = 3, skip = 2):
        super().__init__()
        self.depth = depth
        self.skip = skip
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            # SEBlock(channels),
            # nn.Dropout(0.1),
        ) for i in range(depth)])
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            # nn.Dropout(0.1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        stem = self.res(x)
        for i in range(self.depth):
            x = self.blocks[i](x)
            if i != self.depth-1:
                x = self.relu(x)
        return x + stem

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pooling theo kênh: Max và Avg
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        attn = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        return x * attn
    
class Model(nn.Module):
    def __init__(self, in_channels, numClasses, base_channels, kernel_size, padding, max_depth, skip = 2, num_blocks=3):
        super().__init__()
        self.max_depth = max_depth
        self.skip = skip

        self.num_blocks = num_blocks
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size, padding=padding, stride = 2, bias=False),
            nn.BatchNorm2d(base_channels),
            SEBlock(base_channels),
            SpatialAttention(),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([nn.Sequential(
            ResidualBlock(base_channels, kernel_size, padding, max_depth, self.skip),
            nn.AvgPool2d(2),
            nn.ReLU(inplace=True),
        ) for i in range(num_blocks)])
        self.tail = nn.Sequential(
            ResidualBlock(base_channels, kernel_size, padding, max_depth, self.skip),
            # SEBlock(base_channels),
            # SpatialAttention(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(base_channels, numClasses)
        )
        
    def forward(self, x):
        x = self.stem(x)
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        x = self.tail(x)
        x = torch.flatten(x, 1)
        out = self.fc1(x)
        
        return out, x
import torchvision.models as models
class Backbone(nn.Module):
    def __init__(self, numClasses):
        super().__init__()
        # Load ViT-Ti pretrained
        backbone = models.resnet18(pretrained=True)
        # backbone.fc = torch.nn.Linear(backbone.fc.in_features, embed_dim)
        feat_size = 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
  
        self.fc = nn.Linear(feat_size, numClasses)
    def forward(self, x):
        features = self.backbone(x)          # [B, 192]
        # projected = self.projection(features)  # [B, 512]
        # projected = projected / projected.norm(dim=-1, keepdim=True)  # normalize
        # features = features / features.norm(dim=-1, keepdim=True)  # normalize
        return self.fc(features), features
