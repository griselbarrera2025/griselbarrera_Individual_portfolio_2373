#!/usr/bin/env python3
import argparse, random
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.7)
    args = p.parse_args()
    val = random.random()
    print(f"Sensor reading: {val:.3f}")
    print("ALERT!" if val > args.threshold else "OK")
