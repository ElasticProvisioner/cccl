#!/usr/bin/env python
"""
Compare Python cuda.compute vs C++ CCCL benchmark performance.

Usage:
    # Compare by benchmark name (looks in results/ directory)
    python python_vs_cpp_summary.py -b fill
    python python_vs_cpp_summary.py -b fill -d 0
    python python_vs_cpp_summary.py -b fill -d 0 -o results/fill_comparison.txt

    # Compare all supported benchmarks
    python python_vs_cpp_summary.py

    # Legacy: Compare specific files
    python python_vs_cpp_summary.py results/fill_py.json results/fill_cpp.json
"""

import argparse
import math
import sys
from pathlib import Path

try:
    import tabulate
except ImportError:
    print("Error: tabulate not installed. Run: pip install tabulate")
    sys.exit(1)

import utils

# Supported benchmarks (must match run_benchmarks.sh)
SUPPORTED_BENCHMARKS = [
    "fill",
    # Add more as implemented:
    # "babelstream",
    # "reduce_sum",
    # "scan_exclusive_sum",
    # "histogram_even",
    # "select_if",
    # "radix_sort_keys",
    # "segmented_reduce_sum",
    # "unique_by_key",
]


def extract_measurements(results):
    """Extract all state measurements with GPU and CPU time."""
    measurements = []

    for benchmark in results.get("benchmarks", []):
        for state in benchmark.get("states", []):
            if state.get("is_skipped"):
                continue

            # Get axis values
            # Normalize axis names by removing nvbench tags like {io}, {ct}
            axes = {}
            for ax in state.get("axis_values", []):
                name = ax["name"]
                # Remove nvbench tags: Elements{io} → Elements, T{ct} → T
                if "{" in name:
                    name = name.split("{")[0]
                axes[name] = ax["value"]

            # Get GPU and CPU time from summaries
            gpu_time = None
            cpu_time = None
            for summary in state.get("summaries", []):
                summary_name = summary.get("name", "")
                if (
                    "GPU Time" in summary_name
                    and "Min" not in summary_name
                    and "Max" not in summary_name
                ):
                    for data in summary.get("data", []):
                        if data["name"] == "value":
                            gpu_time = float(data["value"])
                            break
                elif summary_name == "CPU Time":  # Mean CPU time
                    for data in summary.get("data", []):
                        if data["name"] == "value":
                            cpu_time = float(data["value"])
                            break

            if gpu_time:
                measurements.append(
                    {
                        "device": state["device"],
                        "axes": axes,
                        "gpu_time": gpu_time,
                        "cpu_time": cpu_time,  # May be None
                    }
                )

    return measurements


def format_duration(seconds):
    """Format time using nvbench conventions."""
    if seconds >= 1:
        return "%0.3f s" % seconds
    elif seconds >= 1e-3:
        return "%0.3f ms" % (seconds * 1e3)
    elif seconds >= 1e-6:
        return "%0.3f us" % (seconds * 1e6)
    else:
        return "%0.3f us" % (seconds * 1e6)


def format_percentage(value):
    """Format percentage value."""
    return "%0.2f%%" % (value * 100.0)


def _print_comparison_table(py_measurements, cpp_measurements):
    """Print comparison table for given measurements."""
    # Group by element size
    element_sizes = sorted(
        set(int(m["axes"].get("Elements", 0)) for m in py_measurements)
    )

    # Build table data with CPU time for interpreter overhead
    table_data = []
    headers = [
        "Elements",
        "C++ GPU",
        "Py GPU",
        "% Slower",
        "C++ CPU",
        "Py CPU",
        "CPU Ovhd",
    ]

    for size in element_sizes:
        py_times = [
            m["gpu_time"]
            for m in py_measurements
            if int(m["axes"].get("Elements", 0)) == size
        ]
        cpp_times = [
            m["gpu_time"]
            for m in cpp_measurements
            if int(m["axes"].get("Elements", 0)) == size
        ]

        py_cpu_times = [
            m["cpu_time"]
            for m in py_measurements
            if int(m["axes"].get("Elements", 0)) == size
            and m.get("cpu_time") is not None
        ]
        cpp_cpu_times = [
            m["cpu_time"]
            for m in cpp_measurements
            if int(m["axes"].get("Elements", 0)) == size
            and m.get("cpu_time") is not None
        ]

        if not py_times or not cpp_times:
            continue

        # Average if multiple measurements
        py_avg = sum(py_times) / len(py_times)
        cpp_avg = sum(cpp_times) / len(cpp_times)

        overhead = py_avg - cpp_avg
        pct_slower = (overhead / cpp_avg) * 100.0

        # Format element size as power of 2
        log2_size = int(math.log2(size))
        size_str = f"2^{log2_size}"

        if py_cpu_times and cpp_cpu_times:
            # Show CPU time comparison for Python interpreter overhead
            py_cpu_avg = sum(py_cpu_times) / len(py_cpu_times)
            cpp_cpu_avg = sum(cpp_cpu_times) / len(cpp_cpu_times)
            cpu_overhead = py_cpu_avg - cpp_cpu_avg

            table_data.append(
                [
                    size_str,
                    format_duration(cpp_avg),
                    format_duration(py_avg),
                    format_percentage(pct_slower / 100.0),
                    format_duration(cpp_cpu_avg),
                    format_duration(py_cpu_avg),
                    format_duration(cpu_overhead),
                ]
            )
        else:
            # Fallback if CPU times not available
            table_data.append(
                [
                    size_str,
                    format_duration(cpp_avg),
                    format_duration(py_avg),
                    format_percentage(pct_slower / 100.0),
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )

    # Print table using tabulate
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="github"))


def compare_benchmark(py_path, cpp_path, device=None, output_file=None):
    """Compare Python vs C++ benchmark results."""
    if not py_path.exists():
        print(f"Error: Python results not found: {py_path}")
        return False
    if not cpp_path.exists():
        print(f"Error: C++ results not found: {cpp_path}")
        return False

    py_results = utils.read_file(py_path)
    cpp_results = utils.read_file(cpp_path)

    py_measurements = extract_measurements(py_results)
    cpp_measurements = extract_measurements(cpp_results)

    # Filter by device if requested
    if device is not None:
        py_measurements = [m for m in py_measurements if m["device"] == device]
        cpp_measurements = [m for m in cpp_measurements if m["device"] == device]

    if not py_measurements or not cpp_measurements:
        print("No matching measurements found!")
        return False

    # Capture output if writing to file
    output_lines = []

    def output(line=""):
        print(line)
        output_lines.append(line)

    # Get benchmark name
    bench_name = py_results.get("benchmarks", [{}])[0].get("name", "unknown")

    output(f"# {bench_name}")
    output()
    output("GPU Time: Mean GPU execution time (cold start, pure kernel)")
    output("  CUDA events (nvbench tag: nv/cold/time/gpu/mean)")
    output("CPU Time: Mean CPU (host) latency")
    output("  Host clock (nvbench tag: nv/cold/time/cpu/mean)")
    output()

    # Get unique devices
    py_devices = sorted(set(m["device"] for m in py_measurements))
    cpp_devices = sorted(set(m["device"] for m in cpp_measurements))

    for device_id in py_devices:
        if device_id not in cpp_devices:
            continue

        # Get device info
        py_device = utils.find_device_by_id(device_id, py_results.get("devices", []))
        device_name = py_device["name"] if py_device else f"Device {device_id}"

        output(f"## [{device_id}] {device_name}")
        output()

        # Filter measurements for this device
        py_device_measurements = [
            m for m in py_measurements if m["device"] == device_id
        ]
        cpp_device_measurements = [
            m for m in cpp_measurements if m["device"] == device_id
        ]

        # Get unique types (if present)
        types = sorted(set(m["axes"].get("T", "") for m in py_device_measurements))
        has_types = any(types)

        # Group by type (if multi-type benchmark) or just element size
        if has_types and len(types) > 1:
            # Multi-type benchmark: show separate table per type
            for type_str in types:
                if not type_str:
                    continue

                output(f"### Type: {type_str}")
                output()

                py_type_measurements = [
                    m for m in py_device_measurements if m["axes"].get("T") == type_str
                ]
                cpp_type_measurements = [
                    m for m in cpp_device_measurements if m["axes"].get("T") == type_str
                ]

                _print_comparison_table(py_type_measurements, cpp_type_measurements)
                output()
        else:
            # Single-type or no type axis: show one table
            _print_comparison_table(py_device_measurements, cpp_device_measurements)
            output()

    # Write to file if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(output_lines))
        print(f"\nComparison saved to: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python vs C++ CCCL benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare by benchmark name
  %(prog)s -b fill
  %(prog)s -b fill -d 0
  %(prog)s -b fill -d 0 -o results/fill_comparison.txt

  # Compare all supported benchmarks
  %(prog)s

  # Legacy: Compare specific files
  %(prog)s results/fill_py.json results/fill_cpp.json --device 0

Supported benchmarks: %(benchmarks)s
"""
        % {
            "prog": "python_vs_cpp_summary.py",
            "benchmarks": ", ".join(SUPPORTED_BENCHMARKS),
        },
    )

    # New interface: benchmark name
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Benchmark name (e.g., fill). If not specified, compares all supported benchmarks.",
    )
    parser.add_argument(
        "-d", "--device", type=int, help="Filter by GPU device ID (default: all)"
    )
    parser.add_argument("-o", "--output", type=Path, help="Save comparison to file")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results",
        help="Results directory (default: ../results)",
    )

    # Legacy interface: positional args for file paths
    parser.add_argument(
        "python_json", nargs="?", help="Python benchmark results JSON (legacy)"
    )
    parser.add_argument(
        "cpp_json", nargs="?", help="C++ benchmark results JSON (legacy)"
    )

    args = parser.parse_args()

    # Determine mode: legacy (positional args) or new (-b flag)
    if args.python_json and args.cpp_json:
        # Legacy mode: explicit file paths
        py_path = Path(args.python_json)
        cpp_path = Path(args.cpp_json)
        compare_benchmark(py_path, cpp_path, args.device, args.output)

    elif args.benchmark:
        # New mode: single benchmark by name
        if args.benchmark not in SUPPORTED_BENCHMARKS:
            print(
                f"Error: Unknown benchmark '{args.benchmark}'. "
                f"Supported: {', '.join(SUPPORTED_BENCHMARKS)}"
            )
            sys.exit(1)

        py_path = args.results_dir / f"{args.benchmark}_py.json"
        cpp_path = args.results_dir / f"{args.benchmark}_cpp.json"
        output_path = (
            args.output or args.results_dir / f"{args.benchmark}_comparison.txt"
        )

        if not compare_benchmark(py_path, cpp_path, args.device, output_path):
            sys.exit(1)

    else:
        # Default: compare all supported benchmarks
        print("Comparing all supported benchmarks...\n")
        any_success = False

        for bench in SUPPORTED_BENCHMARKS:
            py_path = args.results_dir / f"{bench}_py.json"
            cpp_path = args.results_dir / f"{bench}_cpp.json"

            if not py_path.exists() or not cpp_path.exists():
                print(f"Skipping {bench}: results not found")
                print(f"  Run: ./run_benchmarks.sh -b {bench}")
                print()
                continue

            output_path = args.results_dir / f"{bench}_comparison.txt"
            print(f"=" * 72)
            print(f"Benchmark: {bench}")
            print(f"=" * 72)
            print()

            if compare_benchmark(py_path, cpp_path, args.device, output_path):
                any_success = True
            print()

        if not any_success:
            print("No benchmark results found. Run benchmarks first:")
            print("  ./run_benchmarks.sh -b fill")
            sys.exit(1)


if __name__ == "__main__":
    main()
