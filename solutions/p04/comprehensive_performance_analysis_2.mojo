"""
Comprehensive performance analysis script for add_10_2d implementations using Mojo's official benchmark module.

This script tests multiple matrix sizes to analyze performance scaling
and identify the crossover point where GPU implementations become advantageous.
"""

from benchmark import Bench, Bencher, BenchId, BenchConfig, Unit
from memory import UnsafePointer
from gpu.host import DeviceContext
from gpu import thread_idx
from layout import Layout, LayoutTensor

alias dtype = DType.float32


fn add_10_2d_cpu(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """CPU-only implementation of add_10_2d function."""
    for row in range(size):
        for col in range(size):
            index = row * size + col
            output[index] = a[index] + 10.0


fn add_10_2d_gpu_unsafe(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """GPU kernel using UnsafePointer."""
    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0


fn add_10_2d_layout_tensor(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """GPU kernel using LayoutTensor approach but with UnsafePointer for flexibility.
    """
    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:
        output[row * size + col] = a[row * size + col] + 10.0


@parameter
@always_inline
fn benchmark_cpu_for_size(mut bencher: Bencher, size: Int) raises:
    """Benchmark CPU implementation for given size using benchmark module."""
    # Pre-allocate memory outside the benchmark loop
    var input_data = UnsafePointer[Scalar[dtype]].alloc(size * size)
    var output_data = UnsafePointer[Scalar[dtype]].alloc(size * size)

    # Initialize input data once
    for i in range(size * size):
        input_data[i] = Scalar[dtype](i)

    @parameter
    @always_inline
    fn run_cpu_benchmark():
        # Reset output data
        for i in range(size * size):
            output_data[i] = Scalar[dtype](0.0)
        # Core computation only
        add_10_2d_cpu(output_data, input_data, size)

    bencher.iter[run_cpu_benchmark]()

    # Clean up memory
    input_data.free()
    output_data.free()


@parameter
@always_inline
fn benchmark_gpu_unsafe_for_size(
    mut bencher: Bencher,
    gpu_data: (
        UnsafePointer[Scalar[dtype]],
        UnsafePointer[Scalar[dtype]],
        Int,
        Int,
        Int,
    ),
) raises:
    """Benchmark GPU UnsafePointer implementation for given size using benchmark module.
    """
    var out_ptr = gpu_data[0]
    var a_ptr = gpu_data[1]
    var size = gpu_data[2]
    var blocks_needed = gpu_data[3]
    var block_size = gpu_data[4]

    @parameter
    @always_inline
    fn kernel_launch_unsafe(ctx: DeviceContext) raises:
        # Core computation only - launch GPU kernel
        ctx.enqueue_function[add_10_2d_gpu_unsafe](
            out_ptr,
            a_ptr,
            size,
            grid_dim=blocks_needed,
            block_dim=(block_size, block_size),
        )

    var bench_ctx = DeviceContext()
    bencher.iter_custom[kernel_launch_unsafe](bench_ctx)


@parameter
@always_inline
fn benchmark_gpu_layout_tensor_for_size(
    mut bencher: Bencher,
    gpu_data: (
        UnsafePointer[Scalar[dtype]],
        UnsafePointer[Scalar[dtype]],
        Int,
        Int,
        Int,
    ),
) raises:
    """Benchmark GPU LayoutTensor implementation for given size using benchmark module.
    """
    var out_ptr = gpu_data[0]
    var a_ptr = gpu_data[1]
    var size = gpu_data[2]
    var blocks_needed = gpu_data[3]
    var block_size = gpu_data[4]

    @parameter
    @always_inline
    fn kernel_launch_layout_tensor(ctx: DeviceContext) raises:
        # Core computation only - launch GPU kernel
        ctx.enqueue_function[add_10_2d_layout_tensor](
            out_ptr,
            a_ptr,
            size,
            grid_dim=blocks_needed,
            block_dim=(block_size, block_size),
        )

    var bench_ctx = DeviceContext()
    bencher.iter_custom[kernel_launch_layout_tensor](bench_ctx)


fn run_benchmarks_for_size(
    size: Int,
) raises -> (Float64, Float64, Float64, Bool, Bool):
    """Run benchmarks for a specific matrix size and return timing results."""
    print("Testing size", size, "x", size, "...")

    # Create benchmark instance
    var bench = Bench(BenchConfig())

    # Calculate grid dimensions for GPU benchmarks
    var block_size = 16
    var blocks_needed = (size + block_size - 1) // block_size

    # Setup GPU buffers once for both GPU benchmarks
    var gpu_unsafe_success = True
    var gpu_layout_success = True
    var out_ptr: UnsafePointer[Scalar[dtype]]
    var a_ptr: UnsafePointer[Scalar[dtype]]

    try:
        with DeviceContext() as ctx:
            var out = ctx.enqueue_create_buffer[dtype](
                size * size
            ).enqueue_fill(0)
            var a = ctx.enqueue_create_buffer[dtype](size * size).enqueue_fill(
                0
            )

            # Initialize input data
            with a.map_to_host() as a_host:
                for i in range(size * size):
                    a_host[i] = i

            out_ptr = out.unsafe_ptr()
            a_ptr = a.unsafe_ptr()

            # Benchmark CPU implementation
            bench.bench_with_input[Int, benchmark_cpu_for_size](
                BenchId("add_10_2d_cpu", String("size_") + String(size)), size
            )

            # Prepare GPU data tuple
            var gpu_data = (out_ptr, a_ptr, size, blocks_needed, block_size)

            # Benchmark GPU UnsafePointer implementation
            bench.bench_with_input[
                (
                    UnsafePointer[Scalar[dtype]],
                    UnsafePointer[Scalar[dtype]],
                    Int,
                    Int,
                    Int,
                ),
                benchmark_gpu_unsafe_for_size,
            ](
                BenchId("add_10_2d_gpu_unsafe", String("size_") + String(size)),
                gpu_data,
            )

            # Benchmark GPU LayoutTensor implementation
            bench.bench_with_input[
                (
                    UnsafePointer[Scalar[dtype]],
                    UnsafePointer[Scalar[dtype]],
                    Int,
                    Int,
                    Int,
                ),
                benchmark_gpu_layout_tensor_for_size,
            ](
                BenchId("add_10_2d_gpu_layout", String("size_") + String(size)),
                gpu_data,
            )

            ctx.synchronize()
    except:
        gpu_unsafe_success = False
        gpu_layout_success = False
        print("  GPU benchmark setup failed for size", size)

    # Print benchmark results
    print(bench)

    # Extract timing data
    var cpu_time: Float64 = 0.0
    var gpu_unsafe_time: Float64 = 0.0
    var gpu_layout_time: Float64 = 0.0

    for info in bench.info_vec:
        var name = info.name
        var time_ms = info.result.mean("ms")

        if name.startswith("add_10_2d_cpu"):
            cpu_time = time_ms
        elif name.startswith("add_10_2d_gpu_unsafe"):
            gpu_unsafe_time = time_ms
        elif name.startswith("add_10_2d_gpu_layout"):
            gpu_layout_time = time_ms

    return (
        cpu_time,
        gpu_unsafe_time,
        gpu_layout_time,
        gpu_unsafe_success,
        gpu_layout_success,
    )


fn print_performance_analysis(
    size: Int,
    cpu_time: Float64,
    gpu_unsafe_time: Float64,
    gpu_unsafe_success: Bool,
    gpu_layout_time: Float64,
    gpu_layout_success: Bool,
):
    """Print detailed performance analysis for a given size."""
    elements = size * size
    print(
        "\n=== Performance Analysis for",
        size,
        "x",
        size,
        "matrix (" + String(elements) + " elements) ===",
    )
    print("Average execution times:")
    print("  CPU Implementation:        ", cpu_time, "ms")

    if gpu_unsafe_success:
        print("  GPU UnsafePointer:         ", gpu_unsafe_time, "ms")
    else:
        print("  GPU UnsafePointer:         FAILED")

    if gpu_layout_success:
        print("  GPU LayoutTensor:          ", gpu_layout_time, "ms")
    else:
        print("  GPU LayoutTensor:          FAILED")

    # Calculate speedups and advantages only if implementations succeeded
    if gpu_unsafe_success and cpu_time > 0 and gpu_unsafe_time > 0:
        speedup = cpu_time / gpu_unsafe_time
        if speedup > 1.0:
            print("GPU UnsafePointer vs CPU speedup:", speedup, "x")
        else:
            print("CPU vs GPU UnsafePointer advantage:", 1.0 / speedup, "x")

    if gpu_layout_success and cpu_time > 0 and gpu_layout_time > 0:
        speedup = cpu_time / gpu_layout_time
        if speedup > 1.0:
            print("GPU LayoutTensor vs CPU speedup:", speedup, "x")
        else:
            print("CPU vs GPU LayoutTensor advantage:", 1.0 / speedup, "x")

    # Compare GPU implementations
    if (
        gpu_unsafe_success
        and gpu_layout_success
        and gpu_unsafe_time > 0
        and gpu_layout_time > 0
    ):
        if gpu_layout_time < gpu_unsafe_time:
            speedup = gpu_unsafe_time / gpu_layout_time
            print(
                "GPU LayoutTensor is",
                speedup,
                "x faster than GPU UnsafePointer",
            )
        else:
            speedup = gpu_layout_time / gpu_unsafe_time
            print(
                "GPU UnsafePointer is",
                speedup,
                "x faster than GPU LayoutTensor",
            )

    # Identify fastest implementation
    var fastest_time = cpu_time
    var fastest_name = String("CPU")

    if (
        gpu_unsafe_success
        and gpu_unsafe_time > 0
        and gpu_unsafe_time < fastest_time
    ):
        fastest_time = gpu_unsafe_time
        fastest_name = String("GPU UnsafePointer")

    if (
        gpu_layout_success
        and gpu_layout_time > 0
        and gpu_layout_time < fastest_time
    ):
        fastest_time = gpu_layout_time
        fastest_name = String("GPU LayoutTensor")

    print(
        "Fastest implementation:",
        fastest_name,
        "with",
        fastest_time,
        "ms average",
    )

    # Calculate throughput
    if cpu_time > 0:
        cpu_throughput = Float64(elements) / cpu_time / 1000.0
        print("CPU Throughput:", cpu_throughput, "M elements/ms")

    if gpu_unsafe_success and gpu_unsafe_time > 0:
        gpu_unsafe_throughput = Float64(elements) / gpu_unsafe_time / 1000.0
        print(
            "GPU UnsafePointer Throughput:",
            gpu_unsafe_throughput,
            "M elements/ms",
        )

    if gpu_layout_success and gpu_layout_time > 0:
        gpu_layout_throughput = Float64(elements) / gpu_layout_time / 1000.0
        print(
            "GPU LayoutTensor Throughput:",
            gpu_layout_throughput,
            "M elements/ms",
        )


def main():
    """Main comprehensive performance analysis function using Mojo's official benchmark module.
    """
    print(
        "=== Comprehensive Performance Analysis: add_10_2d Implementations ==="
    )
    print("Testing multiple matrix sizes to find GPU vs CPU crossover point")
    print("Using Mojo's official benchmark module for accurate timing")
    print()

    # Test different matrix sizes - start small and go larger
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print("Performance Results:")
    print("===================")

    crossover_found = False
    crossover_size = 0
    gpu_layout_crossover_found = False
    gpu_layout_crossover_size = 0

    for size in sizes:
        try:
            results = run_benchmarks_for_size(size)
            cpu_time = results[0]
            gpu_unsafe_time = results[1]
            gpu_layout_time = results[2]
            gpu_unsafe_success = results[3]
            gpu_layout_success = results[4]

            print_performance_analysis(
                size,
                cpu_time,
                gpu_unsafe_time,
                gpu_unsafe_success,
                gpu_layout_time,
                gpu_layout_success,
            )

            # Check for crossover point (either GPU implementation faster than CPU)
            if not crossover_found:
                if (
                    gpu_unsafe_success
                    and gpu_unsafe_time > 0
                    and gpu_unsafe_time < cpu_time
                ) or (
                    gpu_layout_success
                    and gpu_layout_time > 0
                    and gpu_layout_time < cpu_time
                ):
                    crossover_found = True
                    crossover_size = size
                    print(
                        "🎯 CROSSOVER POINT FOUND! GPU becomes faster at size",
                        size,
                        "x",
                        size,
                    )

            # Check for GPU LayoutTensor vs UnsafePointer crossover point
            if not gpu_layout_crossover_found:
                if (
                    gpu_unsafe_success
                    and gpu_layout_success
                    and gpu_unsafe_time > 0
                    and gpu_layout_time > 0
                    and gpu_layout_time < gpu_unsafe_time
                ):
                    gpu_layout_crossover_found = True
                    gpu_layout_crossover_size = size
                    print(
                        (
                            "🔄 GPU LAYOUT CROSSOVER FOUND! LayoutTensor becomes"
                            " faster than UnsafePointer at size"
                        ),
                        size,
                        "x",
                        size,
                    )

        except:
            print("❌ Benchmark failed for size", size, "x", size)

        print()

    print("=== Analysis Summary ===")
    print("Crossover Points:")
    if crossover_found:
        print(
            "✅ CPU to GPU crossover:",
            crossover_size,
            "x",
            crossover_size,
            "matrix (GPU becomes faster than CPU)",
        )
    else:
        print("❌ No CPU to GPU crossover point found in tested range")

    if gpu_layout_crossover_found:
        print(
            "✅ GPU UnsafePointer to GPU LayoutTensor crossover:",
            gpu_layout_crossover_size,
            "x",
            gpu_layout_crossover_size,
            "matrix (LayoutTensor becomes faster than UnsafePointer)",
        )
    else:
        print(
            "❌ No GPU UnsafePointer to GPU LayoutTensor crossover point found"
        )

    print("📊 Recommendations:")
    if crossover_found and gpu_layout_crossover_found:
        # Both crossover points found - need to handle the logic carefully
        if gpu_layout_crossover_size < crossover_size:
            # LayoutTensor becomes faster than UnsafePointer before GPU becomes faster than CPU
            print(
                "   - Use CPU implementation for matrices 2 x 2 to",
                crossover_size - 1,
                "x",
                crossover_size - 1,
            )
            print(
                "   - Use GPU LayoutTensor for matrices",
                crossover_size,
                "x",
                crossover_size,
                "and larger",
            )
        else:
            # GPU becomes faster than CPU before LayoutTensor becomes faster than UnsafePointer
            print(
                "   - Use CPU implementation for matrices 2 x 2 to",
                crossover_size - 1,
                "x",
                crossover_size - 1,
            )
            print(
                "   - Use GPU UnsafePointer for matrices",
                crossover_size,
                "x",
                crossover_size,
                "to",
                gpu_layout_crossover_size - 1,
                "x",
                gpu_layout_crossover_size - 1,
            )
            print(
                "   - Use GPU LayoutTensor for matrices",
                gpu_layout_crossover_size,
                "x",
                gpu_layout_crossover_size,
                "and larger",
            )
    elif crossover_found:
        # Only CPU to GPU crossover found
        if gpu_layout_crossover_found:
            # This case shouldn't happen logically, but handle it
            print(
                "   - Use CPU implementation for matrices 2 x 2 to",
                crossover_size - 1,
                "x",
                crossover_size - 1,
            )
            print(
                "   - Use GPU LayoutTensor for matrices",
                crossover_size,
                "x",
                crossover_size,
                "and larger (LayoutTensor preferred)",
            )
        else:
            print(
                "   - Use CPU implementation for matrices 2 x 2 to",
                crossover_size - 1,
                "x",
                crossover_size - 1,
            )
            print(
                "   - Use GPU UnsafePointer for matrices",
                crossover_size,
                "x",
                crossover_size,
                "and larger",
            )
    elif gpu_layout_crossover_found:
        # Only GPU LayoutTensor crossover found (GPU never beats CPU)
        print("   - Use CPU implementation for all tested matrix sizes")
        print(
            "   - Among GPU implementations, prefer LayoutTensor for matrices",
            gpu_layout_crossover_size,
            "x",
            gpu_layout_crossover_size,
            "and larger",
        )
    else:
        print("   - Use CPU implementation for all tested matrix sizes")
        print("   - GPU overhead dominates for these matrix sizes")
        print(
            "   - Consider testing larger matrices or more complex operations"
        )

    print()
    print("=== Comprehensive Performance Analysis Complete ===")
