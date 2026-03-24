#!/usr/bin/env python
"""
Extract annual return from RARS backtest output.
Usage: python scripts/extract_backtest_metric.py < backtest_output.txt
"""
import sys
import re

def main():
    # Read all input
    output = sys.stdin.read()

    # Extract annual return (supports both Chinese and English output)
    match = re.search(r'年化收益率:\s*([-\d.]+)%', output) or \
            re.search(r'Annual Return:\s*([-\d.]+)%', output) or \
            re.search(r'annual_return.*?([-\d.]+)', output, re.IGNORECASE)

    if match:
        value = float(match.group(1))
        print(f"{value:.4f}")
        return 0
    else:
        # Fallback: try to find any percentage-like number in the output
        numbers = re.findall(r'[-+]?\d*\.\d+%?', output)
        if numbers:
            print(f"ERROR: Could not find '年化收益率' line. Found numbers: {numbers[:5]}")
        else:
            print("ERROR: No percentage found in output")
        return 1

if __name__ == '__main__':
    sys.exit(main())
