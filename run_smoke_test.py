#!/usr/bin/env python3
"""Quick BSE + CGCL pipeline check on datasets/test.csv.

Runs one unsupervised BSE epoch and one epoch in each active CGCL stage.
The four-cluster setting is a smoke-test parameter, not a training label.
No HCRL or legacy plotting/comparison is run.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:5',
                        help='PyTorch device, for example cuda:5 or cpu')
    parser.add_argument('--backbone', type=Path,
                        default=Path('/mnt/data2/pengy/chinese-roberta-wwm-ext'))
    args = parser.parse_args()

    dataset = ROOT / 'datasets' / 'test.csv'
    if not dataset.is_file():
        parser.error(f'Test dataset is missing: {dataset}')
    if not args.backbone.is_dir():
        parser.error(f'Backbone directory is missing: {args.backbone}')

    command = [
        sys.executable, '-B', str(ROOT / 'main_ACRL_可视化.py'),
        '--csv', str(dataset), '--backbone', str(args.backbone),
        '--device', args.device, '--seed', '42',
        '--epochs-bse', '1', '--epochs-stage1', '0',
        '--epochs-stage2', '1', '--epochs-stage3', '1',
        '--batch-size', '16', '--fixed-k', '4',
        '--pseudo-update-interval', '1',
        '--stage2-fine-weight', '1.0',
        '--skip-comparison',
    ]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
