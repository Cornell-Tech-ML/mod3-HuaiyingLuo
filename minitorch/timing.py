import minitorch
import time
import numpy as np
import matplotlib.pyplot as plt


FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Run a matrix multiplication benchmark."""
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    _ = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    sizes = [64, 128, 256, 512, 1024]
    for size in sizes:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

    # Plotting
    fast_means = [times[size]["fast"] for size in sizes]
    cuda_means = [times[size]["gpu"] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_means, label="FastOps", marker="o")
    plt.plot(sizes, cuda_means, label="CudaOps", marker="o")
    plt.xlabel("Matrix Size")
    plt.ylabel("Average Time (s)")
    plt.title("Matrix Multiplication Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("matrix_multiplication_performance.png")
    plt.show()
