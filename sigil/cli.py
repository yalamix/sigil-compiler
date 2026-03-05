# sigil/cli.py

import argparse
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# Ensure project root is on sys.path so 'tests' is importable
# regardless of where the command is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests(module_name):
    """Dispatch to the appropriate test module."""
    if module_name == 'segmentation':
        from tests.test_geometry.test_segmentation import run_all
        run_all()
    else:
        print(f"Unknown test module: '{module_name}'")
        print("Available: segmentation")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(prog='sigil')
    subparsers = parser.add_subparsers(dest='command')

    test_parser = subparsers.add_parser('test', help='Run module tests')
    test_parser.add_argument('module', help='Module to test')

    args = parser.parse_args()

    if args.command == 'test':
        run_tests(args.module)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()