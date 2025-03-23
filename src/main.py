import argparse
import sys

from solver import run_optimal_solver, run_single_solver
from recalc_solver import run_recalc_solver

def print_help():
    help_text = """
Usage: script.py <mode> --data <file_path> [--debug] [--random]

Modes:
  solve         Find paths for provided drones, coords, etc. Check data/example_solve.json for additional info
  optimal       Find paths and optimal amount of drones for a given coords
  recalc        Find paths for already dispatched drones

Options:
  --data    Path to the data file (required)
  --debug   Enable debug output (optional)
  --random  Uses random data for recalc mode (optional)
  --help    Show this help message and exit
"""
    print(help_text)

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Run a mode with a data file and optional debug output.", add_help=False)
    parser.add_argument("mode", choices=["solve", "optimal", "recalc"], help="Mode of operation")
    parser.add_argument("--data", required=True, help="Path to the data file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--random", action="store_true", help="Uses random data for recalc mode")

    args = parser.parse_args()

    mode_function_map = {
        "solve": run_single_solver,
        "optimal": run_optimal_solver,
        "recalc": run_recalc_solver,
    }

    mode_func = mode_function_map.get(args.mode)
    if args.debug:
        print(f"[DEBUG] Running {args.mode}() with data={args.data} and random={args.random}")
    mode_func(args.data, {'debug': args.debug, 'random': args.random})

if __name__ == "__main__":
    main()
