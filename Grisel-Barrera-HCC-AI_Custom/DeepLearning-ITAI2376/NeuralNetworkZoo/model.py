#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim=32, hid=16, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class TinyCNN(nn.Module):
    def __init__(self, in_ch=1, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU()
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()
    if args.demo:
        print("Neural Network Zoo: TinyMLP and TinyCNN are defined and ready.")
