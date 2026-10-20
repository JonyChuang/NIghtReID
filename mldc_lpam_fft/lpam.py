"""Author FFT LPAM, including its non-residual ConvBlock and CLS input.
Source: zhguoqing/A3PFN, archived commit 5f1457f4cd18ee6194771392c3665aec3d22df3c.
The three author classes are preserved without edits.
FFT/IFFT axes, no-conjugation formula, head scaling and sharing are preserved.
This is a single-CLS-token module, not spatial patch attention.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.weights = nn.Parameter(torch.ones(1, heads, 1, 1))  # Initialize weights as (1, heads, 1, 1)

    def forward(self, x):
        B, N, C = x.shape
        queries = self.query(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)
        keys = self.key(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)
        values = self.value(x).view(B, N, self.heads, C // self.heads).transpose(1, 2)

        # Determine the nearest power of 2
        target_size = 2**math.ceil(math.log2(queries.size(-1)))

        # Calculate the size required for padding
        padding_size = target_size - queries.size(-1)

        # Pad using F.pad
        queries_padded = F.pad(queries, (0, padding_size))
        keys_padded = F.pad(keys, (0, padding_size))

        # Convert padded data to float32 before FFT
        queries_padded = queries_padded.to(dtype=torch.float32)
        keys_padded = keys_padded.to(dtype=torch.float32)

        # FFT on the padded data
        queries_fft = torch.fft.fft(queries_padded)
        keys_fft = torch.fft.fft(keys_padded)

        # Dot product of FFTs
        dot = torch.einsum('bhid,bhjd->bhij', queries_fft, keys_fft) * self.scale

        # Inverse FFT and get data of the original size
        dot = torch.fft.ifft(dot).real
        dot = dot[..., :queries.size(-1)]

        attn = F.softmax(dot, dim=-1)

        # Resize self.weights to match the shape of attn
        weights = self.weights.repeat(B, 1, N, N, 1)

        attn = attn.unsqueeze(1)  # Add an extra dimension for the batch size
        attn = attn * weights
        attn = attn.squeeze(1)  # Reduce dimensionality

        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return out

class ConvBlock_FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True) # ffn 全连接层
        self.self_attn = SelfAttention(out_channels)

    def forward(self, x):
        if x.size(1) == self.conv1.in_channels:
            x = self.conv1(x)
        elif x.size(1) == self.conv2.in_channels:
            x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # transpose to (batch_size, sequence_length, num_channels)
        x = self.self_attn(x)
        x = x.permute(0, 2, 1)  # transpose back to (batch_size, num_channels, sequence_length)
        return x

class FeaturePyramid(nn.Module ):
    def __init__(self,feature_dim):
        super().__init__()
        self.conv_block = ConvBlock_FPN(feature_dim ,feature_dim)

    def forward(self,block3,block6,block9,block12):
        f1 = self.conv_block (torch.cat([self.conv_block(block3),self.conv_block (block6)],dim=1))
        f2 = self.conv_block (torch.cat([f1,self.conv_block(block9)],dim = 1))
        f3 = self.conv_block (torch.cat([f2,self.conv_block(block12)],dim = 1))

        f4 = self.conv_block (torch.cat([f1,f2],dim=1))
        f5 = self.conv_block (torch.cat([f4,f3],dim = 1))
        f6 = self.conv_block (torch.cat([f4,f5],dim =1))

        return f3,f5,f6
