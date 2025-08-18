"""Submit CLI - A command-line tool for submitting ML jobs."""

import argparse
import sys


def add_numbers(num1: float, num2: float) -> float:
    """Add two numbers together."""
    return num1 + num2


def main():
    """Main entry point for the submit CLI."""
    parser = argparse.ArgumentParser(
        prog='submit',
        description='Submit ML jobs to remote GPUs (placeholder: currently adds two numbers)',
        epilog='Example: submit 2 3'
    )
    
    # Add positional arguments for the two numbers
    parser.add_argument(
        'num1',
        type=float,
        help='First number to add'
    )
    
    parser.add_argument(
        'num2',
        type=float,
        help='Second number to add'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Perform the addition
    result = add_numbers(args.num1, args.num2)
    
    # Print the result
    print(f"Result: {args.num1} + {args.num2} = {result}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
