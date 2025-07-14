"""
Comprehensive benchmark script for add_10_2d implementations using Mojo's official benchmark module.

This script measures and compares the execution time of three add_10_2d() function implementations:
1. CPU-only version from p04_cpu.mojo
2. GPU implementation from p04.mojo
3. GPU implementation from p04_layout_tensor.mojo

The benchmark uses Mojo's official benchmark module for proper runtime benchmarking and reporting.
"""

from benchmark import Bench, Bencher, BenchId, BenchConfig, Unit
from testing import assert_equal
from memory import UnsafePointer
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

# alias SIZE = 2  # Back to working configuration
alias SIZE = 3  # Back to working configuration

alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)


# CPU implementation (copied from p04_cpu.mojo)
fn add_10_2d_cpu(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """CPU-only implementation of add_10_2d function."""
    for row in range(size):
        for col in range(size):
            var index = row * size + col
            output[index] = a[index] + 10.0


# GPU implementation from p04.mojo
fn add_10_2d_unsafe_ptr(
    output: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    """GPU kernel using UnsafePointer (from p04.mojo)."""
    from gpu import thread_idx

    row = thread_idx.y
    col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0


# GPU implementation from p04_layout_tensor.mojo
fn add_10_2d_layout_tensor(
    output: LayoutTensor[mut=True, dtype, layout],
    a: LayoutTensor[mut=True, dtype, layout],
    size: Int,
):
    """GPU kernel using LayoutTensor (from p04_layout_tensor.mojo)."""
    from gpu import thread_idx

    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:
        output[row, col] = a[row, col] + 10.0


@parameter
@always_inline
fn benchmark_cpu_implementation(mut bencher: Bencher) raises:
    """Benchmark the CPU-only implementation using the benchmark module."""
    # Create test data once outside the benchmark loop
    var input_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)
    var output_data = UnsafePointer[Scalar[dtype]].alloc(SIZE * SIZE)

    # Initialize input data
    for i in range(SIZE * SIZE):
        input_data[i] = Scalar[dtype](i)

    @parameter
    @always_inline
    fn run_cpu_benchmark():
        # Reset output data
        for i in range(SIZE * SIZE):
            output_data[i] = Scalar[dtype](0.0)
        # Core computation only
        add_10_2d_cpu(output_data, input_data, SIZE)

    bencher.iter[run_cpu_benchmark]()

    # Verify correctness after benchmarking
    for i in range(SIZE * SIZE):
        var expected = Scalar[dtype](i + 10)
        assert_equal(output_data[i], expected)

    # Clean up memory
    input_data.free()
    output_data.free()


@parameter
@always_inline
fn benchmark_gpu_unsafe_ptr(
    mut bencher: Bencher,
    input_data: (UnsafePointer[Scalar[dtype]], UnsafePointer[Scalar[dtype]]),
) raises:
    """Benchmark the GPU implementation using UnsafePointer."""
    var out_ptr = input_data[0]
    var a_ptr = input_data[1]

    @parameter
    @always_inline
    fn kernel_launch_unsafe_ptr(ctx: DeviceContext) raises:
        # Core computation only - launch GPU kernel
        ctx.enqueue_function[add_10_2d_unsafe_ptr](
            out_ptr,
            a_ptr,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

    var bench_ctx = DeviceContext()
    bencher.iter_custom[kernel_launch_unsafe_ptr](bench_ctx)


@parameter
@always_inline
fn benchmark_gpu_layout_tensor(
    mut bencher: Bencher,
    input_data: (
        LayoutTensor[dtype, layout, MutableAnyOrigin],
        LayoutTensor[dtype, layout, MutableAnyOrigin],
    ),
) raises:
    """Benchmark the GPU implementation using LayoutTensor."""
    var out_tensor = input_data[0]
    var a_tensor = input_data[1]

    @parameter
    @always_inline
    fn kernel_launch_layout_tensor(ctx: DeviceContext) raises:
        # Core computation only - launch GPU kernel
        ctx.enqueue_function[add_10_2d_layout_tensor](
            out_tensor,
            a_tensor,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

    var bench_ctx = DeviceContext()
    bencher.iter_custom[kernel_launch_layout_tensor](bench_ctx)


def main():
    """Main benchmark function using Mojo's official benchmark module."""
    print("=== Comprehensive add_10_2d Benchmark ===")
    print("Matrix size:", SIZE, "x", SIZE)
    print("GPU blocks per grid:", BLOCKS_PER_GRID)
    print(
        "GPU threads per block:",
        THREADS_PER_BLOCK[0],
        "x",
        THREADS_PER_BLOCK[1],
    )

    # Create benchmark instance
    var bench = Bench(BenchConfig())

    # Benchmark CPU implementation
    bench.bench_function[benchmark_cpu_implementation](
        BenchId("add_10_2d", "cpu")
    )

    # Benchmark GPU implementations
    with DeviceContext() as ctx:
        # Setup GPU buffers for UnsafePointer benchmark
        var out_buf = ctx.enqueue_create_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)
        var a_buf = ctx.enqueue_create_buffer[dtype](SIZE * SIZE).enqueue_fill(
            0
        )

        # Initialize input data
        with a_buf.map_to_host() as a_host:
            for i in range(SIZE * SIZE):
                a_host[i] = i

        var unsafe_ptr_data = (out_buf.unsafe_ptr(), a_buf.unsafe_ptr())

        bench.bench_with_input[
            (UnsafePointer[Scalar[dtype]], UnsafePointer[Scalar[dtype]]),
            benchmark_gpu_unsafe_ptr,
        ](BenchId("add_10_2d", "gpu_unsafe_ptr"), unsafe_ptr_data)

        # Setup GPU buffers for LayoutTensor benchmark
        var out_buf_tensor = ctx.enqueue_create_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)
        var a_buf_tensor = ctx.enqueue_create_buffer[dtype](
            SIZE * SIZE
        ).enqueue_fill(0)

        # Initialize input data
        with a_buf_tensor.map_to_host() as a_host_tensor:
            for i in range(SIZE * SIZE):
                a_host_tensor[i] = i

        var out_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
            out_buf_tensor.unsafe_ptr()
        ).reshape[layout]()
        var a_tensor = LayoutTensor[dtype, layout, MutableAnyOrigin](
            a_buf_tensor.unsafe_ptr()
        ).reshape[layout]()

        var layout_tensor_data = (out_tensor, a_tensor)

        bench.bench_with_input[
            (
                LayoutTensor[dtype, layout, MutableAnyOrigin],
                LayoutTensor[dtype, layout, MutableAnyOrigin],
            ),
            benchmark_gpu_layout_tensor,
        ](BenchId("add_10_2d", "gpu_layout_tensor"), layout_tensor_data)

        # Verify correctness
        ctx.synchronize()

        # Verify UnsafePointer implementation
        with out_buf.map_to_host() as out_host:
            for i in range(SIZE * SIZE):
                var expected = Scalar[dtype](i + 10)
                assert_equal(out_host[i], expected)

        # Verify LayoutTensor implementation
        with out_buf_tensor.map_to_host() as out_host_tensor:
            for i in range(SIZE * SIZE):
                var expected = Scalar[dtype](i + 10)
                assert_equal(out_host_tensor[i], expected)

    # Print benchmark results
    print(bench)

    # Extract timing data for performance comparison
    var cpu_time: Float64 = 0.0
    var gpu_unsafe_time: Float64 = 0.0
    var gpu_layout_time: Float64 = 0.0
    var cpu_found = False
    var gpu_unsafe_found = False
    var gpu_layout_found = False

    # Extract results from benchmark info
    for info in bench.info_vec:
        var name = info.name
        var time_ms = info.result.mean("ms")

        if name == "add_10_2d/cpu":
            cpu_time = time_ms
            cpu_found = True
        elif name == "add_10_2d/gpu_unsafe_ptr":
            gpu_unsafe_time = time_ms
            gpu_unsafe_found = True
        elif name == "add_10_2d/gpu_layout_tensor":
            gpu_layout_time = time_ms
            gpu_layout_found = True

    # Performance comparison analysis
    if cpu_found and gpu_unsafe_found and gpu_layout_found:
        print("\n=== Performance Comparison ===")
        print("Average execution times:")
        print("  CPU:              ", cpu_time, "ms")
        print("  GPU UnsafePointer:", gpu_unsafe_time, "ms")
        print("  GPU LayoutTensor: ", gpu_layout_time, "ms")

        # Calculate speedups relative to CPU (handle edge cases)
        if cpu_time > 0:
            if gpu_unsafe_time > 0:
                var unsafe_speedup = cpu_time / gpu_unsafe_time
                print("GPU UnsafePointer vs CPU speedup:", unsafe_speedup, "x")
            else:
                print("GPU UnsafePointer vs CPU speedup: N/A (zero timing)")

            if gpu_layout_time > 0:
                var layout_speedup = cpu_time / gpu_layout_time
                print("GPU LayoutTensor vs CPU speedup:", layout_speedup, "x")
            else:
                print("GPU LayoutTensor vs CPU speedup: N/A (zero timing)")
        else:
            print("Cannot calculate speedups: CPU timing is zero")

        # Compare GPU implementations
        if gpu_unsafe_time > 0 and gpu_layout_time > 0:
            if gpu_unsafe_time < gpu_layout_time:
                var unsafe_advantage = gpu_layout_time / gpu_unsafe_time
                print(
                    "GPU UnsafePointer is",
                    unsafe_advantage,
                    "x faster than GPU LayoutTensor",
                )
            else:
                var layout_advantage = gpu_unsafe_time / gpu_layout_time
                print(
                    "GPU LayoutTensor is",
                    layout_advantage,
                    "x faster than GPU UnsafePointer",
                )

        # Identify fastest implementation
        var fastest_time = cpu_time
        var fastest_name = String("CPU")

        if gpu_unsafe_time > 0 and gpu_unsafe_time < fastest_time:
            fastest_time = gpu_unsafe_time
            fastest_name = String("GPU UnsafePointer")

        if gpu_layout_time > 0 and gpu_layout_time < fastest_time:
            fastest_time = gpu_layout_time
            fastest_name = String("GPU LayoutTensor")

        print(
            "Fastest implementation:",
            fastest_name,
            "with",
            fastest_time,
            "ms average",
        )
    else:
        print("\n=== Performance Comparison Unavailable ===")
        print("Could not extract timing data for all implementations:")
        if not cpu_found:
            print("  - CPU benchmark not found")
        if not gpu_unsafe_found:
            print("  - GPU UnsafePointer benchmark not found")
        if not gpu_layout_found:
            print("  - GPU LayoutTensor benchmark not found")

    print("\n=== Benchmark Completed ===")
