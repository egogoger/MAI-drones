import argparse
import sys

from solver import run_optimal_solver, run_single_solver
from recalc_solver import run_recalc_solver
from runner import run_solve_tests

def print_help():
    help_text = """
Usage: main.py <mode> --data <file_path> [--debug] [--random]

Modes:
  solve         Find paths for provided drones, coords, etc. Check data/example_solve.json for additional info
  optimal       Find paths and optimal amount of drones for a given coords
  recalc        Find paths for already dispatched drones
  solve-tester  Run multiple tests for "solve" mode

Options:
  --data        Path to the data file (required)
  --debug       Enable debug output (optional)
  --random      Uses random data (optional)
  --help        Show this help message and exit

Solve tester options:
  --coords_n    Max amount of coords (required)
  --start_coords_n    Max amount of coords (required)

Examples:
python3 src/main.py solve --data data/example_solve.json
python3 src/main.py solve --data data/example_solve_random.json --random
python3 src/main.py optimal --data data/example_optimal.json
python3 src/main.py recalc --data data/example_recalc.json
NO_PLOTS=TRUE python3 src/main.py solve-tester --coords_n 10 --start_coords_n 5
"""
    print(help_text)

def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Run a mode with a data file and optional debug output.", add_help=False)
    parser.add_argument("mode", choices=["solve", "optimal", "recalc", "solve-tester"], help="Mode of operation")
    parser.add_argument("--data", help="Path to the data file")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--random", action="store_true", help="Uses random data for recalc mode")
    parser.add_argument("--coords_n", type=int, help="Max amount of coords (required)")
    parser.add_argument("--start_coords_n", type=int, default=2, help="Starting amount of coords")

    args = parser.parse_args()

    mode_function_map = {
        "solve": run_single_solver,
        "optimal": run_optimal_solver,
        "recalc": run_recalc_solver,
        "solve-tester": run_solve_tests,
    }

    mode_func = mode_function_map.get(args.mode)
    if args.debug:
        print(f"[DEBUG] Running {args.mode}() with data={args.data} and random={args.random}")
    if args.mode == "solve-tester":
        if not args.coords_n:
            print("--coords_n arg is required")
            sys.exit(1)
        return mode_func(args.start_coords_n, args.coords_n)
    if not args.data:
        print("--data arg is required")
        sys.exit(1)
    mode_func(args.data, {'debug': args.debug, 'random': args.random})

# TODO: add speed of drones
if __name__ == "__main__":
    main()
