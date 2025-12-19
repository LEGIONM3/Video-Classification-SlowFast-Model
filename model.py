import torch
import torch.nn as nn

ALPHA = 8 

class LateralConnection(nn.Module):
    """
    Fuses Fast pathway features into Slow pathway.
    Transforms Fast features to match Slow features in temporal dimension.
    """
    def __init__(self, fast_channels, slow_channels, alpha=ALPHA):
        super(LateralConnection, self).__init__()
        # 3D Convolution to match duration and channels
        # Kernel size usually (5, 1, 1) or (7, 1, 1) to pool temporal info
        # Stride = (alpha, 1, 1) to match slow temporal dim
        self.conv = nn.Conv3d(fast_channels, slow_channels * 2, kernel_size=(5, 1, 1), stride=(alpha, 1, 1), padding=(2, 0, 0), bias=False)
        
    def forward(self, x_fast):
        return self.conv(x_fast)

class SlowFastNetwork(nn.Module):
    def __init__(self):
        super(SlowFastNetwork, self).__init__()
        
        # --- Fast Pathway (High Frame Rate, Low Channel Capacity) ---
        # Input: (B, 3, 32, 112, 112)
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)) 
        # Output: (B, 8, 32, 28, 28)
        
        self.fast_conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.fast_bn2 = nn.BatchNorm3d(16)
        # Output: (B, 16, 32, 14, 14)
        
        self.fast_conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.fast_bn3 = nn.BatchNorm3d(32)
        # Output: (B, 32, 32, 14, 14)

        # --- Slow Pathway (Low Frame Rate, High Channel Capacity) ---
        # Input: (B, 3, 4, 112, 112)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # Output: (B, 64, 4, 28, 28)
        
        self.slow_conv2 = nn.Conv3d(64 + 16, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.slow_bn2 = nn.BatchNorm3d(128)
        # Output: (B, 128, 4, 14, 14)
        
        self.slow_conv3 = nn.Conv3d(128 + 64, 256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.slow_bn3 = nn.BatchNorm3d(256)
        # Output: (B, 256, 4, 14, 14)
        
        # --- Lateral Connections ---
        # From Fast Stage 1 to Slow Stage 2 input
        self.lateral1 = nn.Conv3d(8, 16, kernel_size=(5, 1, 1), stride=(ALPHA, 1, 1), padding=(2, 0, 0), bias=False)
        
        # From Fast Stage 2 to Slow Stage 3 input
        self.lateral2 = nn.Conv3d(16, 64, kernel_size=(5, 1, 1), stride=(ALPHA, 1, 1), padding=(2, 0, 0), bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 + 256, 2) # Fast final channels (32) + Slow final channels (256)

    def forward(self, slow_input, fast_input):
        # Fast Pathway
        f1 = self.relu(self.fast_bn1(self.fast_conv1(fast_input)))
        f1_p = self.fast_pool1(f1)
        
        # Slow Pathway Step 1
        s1 = self.relu(self.slow_bn1(self.slow_conv1(slow_input)))
        s1_p = self.slow_pool1(s1)
        
        # Lateral Blend 1: Fuse Fast(f1_p) into Slow(s1_p)
        # f1_p: (B, 8, 32, 28, 28) -> lateral -> (B, 16, 4, 28, 28)
        # s1_p: (B, 64, 4, 28, 28)
        # We concatenate features for this simple implementation
        l1 = self.lateral1(f1_p)
        s2_input = torch.cat([s1_p, l1], dim=1) # (64+16) channels
        
        # Fast Stage 2
        f2 = self.relu(self.fast_bn2(self.fast_conv2(f1_p)))
        
        # Slow Stage 2
        s2 = self.relu(self.slow_bn2(self.slow_conv2(s2_input)))
        
        # Lateral Blend 2: Fuse Fast(f2) into Slow(s2)
        # f2: (B, 16, 32, 14, 14) -> lateral -> (B, 64, 4, 14, 14)
        # s2: (B, 128, 4, 14, 14)
        l2 = self.lateral2(f2)
        s3_input = torch.cat([s2, l2], dim=1) # (128+64) channels
        
        # Fast Stage 3
        f3 = self.relu(self.fast_bn3(self.fast_conv3(f2)))
        
        # Slow Stage 3
        s3 = self.relu(self.slow_bn3(self.slow_conv3(s3_input)))
        
        # Global Pooling
        f_out = self.avg_pool(f3).view(f3.size(0), -1) # B, 32
        s_out = self.avg_pool(s3).view(s3.size(0), -1) # B, 256
        
        # Concatenate pathways
        x = torch.cat([s_out, f_out], dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
