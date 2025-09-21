import time
import torch
import triton
import kernels

def create_test_data(batch_size: int, channels: int, device: str = 'cuda'):
    input_tensor = torch.randn(batch_size, channels, device=device, dtype=torch.float32)
    gamma = torch.ones(channels, device=device, dtype=torch.float32)
    beta = torch.zeros(channels, device=device, dtype=torch.float32)
    return input_tensor, gamma, beta

def check_correctness(batch_size: int, channels: int, eps: float = 1e-5):
    print(f"Testing correctness for batch_size={batch_size}, channels={channels}...")
    input_tensor, gamma, beta = create_test_data(batch_size, channels)
    
    output_custom = kernels.batchnorm(input_tensor, gamma, beta, eps)

    bn = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=False, device=input_tensor.device)
    bn.weight.data = gamma
    bn.bias.data = beta
    bn.train()
    with torch.no_grad():
        output_pytorch = bn(input_tensor)

    max_diff = torch.max(torch.abs(output_custom - output_pytorch)).item()
    rel_error = max_diff / (torch.max(torch.abs(output_pytorch)).item() + 1e-8)
    is_correct = max_diff < 1e-3 or rel_error < 1e-3

    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    print(f"  Result: {'✓ PASS' if is_correct else '✗ FAIL'}")
    return is_correct

def benchmark_batch_norm(batch_size: int, channels: int, eps: float = 1e-5, iters: int = 100):
    print(f"Benchmarking batch_size={batch_size}, channels={channels}...")
    input_tensor, gamma, beta = create_test_data(batch_size, channels)
    
    for _ in range(30):
        _ = kernels.batchnorm(input_tensor, gamma, beta, eps)
    torch.cuda.synchronize()

    triton_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = kernels.batchnorm(input_tensor, gamma, beta, eps)
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))
    triton_time = sum(triton_times) / len(triton_times)

    bn = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=False, device=input_tensor.device)
    bn.weight.data = gamma
    bn.bias.data = beta
    bn.train()
    for _ in range(30):
        with torch.no_grad():
            _ = bn(input_tensor)
    torch.cuda.synchronize()

    pytorch_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = bn(input_tensor)
        end.record()
        torch.cuda.synchronize()
        pytorch_times.append(start.elapsed_time(end))
    pytorch_time = sum(pytorch_times) / len(pytorch_times)

    speedup = pytorch_time / triton_time if triton_time > 0 else 0
    bytes_accessed = 2 * batch_size * channels * 4
    bandwidth_gb = bytes_accessed / (triton_time * 1e6)

    print(f"  Triton time: {triton_time:.3f} ms")
    print(f"  PyTorch time: {pytorch_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x {'🚀' if speedup > 1.0 else ''}")
    print(f"  Bandwidth: {bandwidth_gb:.1f} GB/s")

    return {'triton_ms': triton_time, 'pytorch_ms': pytorch_time,
            'speedup': speedup, 'bandwidth_gb': bandwidth_gb}

def run_comprehensive_batch_norm_tests():
    print("=" * 60)
    print("OPTIMIZED BATCH NORMALIZATION (N x C)")
    print("=" * 60)

    test_configs = [
        (32, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
        (512, 2048),
        (1024, 4096),
        (2048, 8192),
        (4096, 16384),
        (8192, 32768),
        (16384, 65536),
    ]

    print("\n1. CORRECTNESS TESTS")
    print("-" * 30)
    all_correct = True
    for batch_size, channels in test_configs[:5]:
        is_correct = check_correctness(batch_size, channels)
        all_correct = all_correct and is_correct
        print()
    print(f"Overall Correctness: {'✓ ALL PASS' if all_correct else '✗ SOME FAILED'}")

    print("\n2. PERFORMANCE BENCHMARKS")
    print("-" * 30)
    results = []
    for batch_size, channels in test_configs:
        result = benchmark_batch_norm(batch_size, channels, iters=100)
        results.append(((batch_size, channels), result))
        print()

    print("\n3. PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"{'Config':<15} {'Triton (ms)':<12} {'PyTorch (ms)':<12} {'Speedup':<10} {'BW (GB/s)':<10}")
    print("-" * 70)

    winning_configs = 0
    for (batch_size, channels), result in results:
        config_str = f"{batch_size}x{channels}"
        speedup_str = f"{result['speedup']:.2f}x"
        if result['speedup'] > 1.0:
            speedup_str += " 🚀"
            winning_configs += 1
        print(f"{config_str:<15} {result['triton_ms']:<12.3f} {result['pytorch_ms']:<12.3f} {speedup_str:<10} {result['bandwidth_gb']:<10.1f}")

    avg_speedup = sum(r[1]['speedup'] for r in results) / len(results)
    geometric_mean = (torch.prod(torch.tensor([r[1]['speedup'] for r in results])) ** (1/len(results))).item()

    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    print(f"Geometric Mean Speedup: {geometric_mean:.2f}x")
    print(f"Winning configurations: {winning_configs}/{len(results)}")

    best = max(results, key=lambda x: x[1]['speedup'])
    worst = min(results, key=lambda x: x[1]['speedup'])
    print(f"\nBest speedup: {best[0][0]}x{best[0][1]} = {best[1]['speedup']:.2f}x")
    print(f"Worst speedup: {worst[0][0]}x{worst[0][1]} = {worst[1]['speedup']:.2f}x")

    print("\n4. ANALYSIS")
    print("-" * 30)
    print("For large batches, PyTorch likely uses:")
    print("- Optimized cuDNN kernels with tensor core utilization")
    print("- Multi-stage reduction algorithms")
    print("- Better memory access patterns for high bandwidth")
    print("\nPossible improvements:")
    print("- Tune fused kernel configs (bigger BLOCK_C on high-end GPUs)")
    print("- Use mixed precision IO (fp16/bf16) with fp32 math to cut bandwidth")
    print("- Consider Welford variance for extreme N if needed")

    return all_correct, results

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires GPU.")
    else:
        print(f"Running on {torch.cuda.get_device_name()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Triton version: {triton.__version__}\n")

        print("Quick Correctness Check:")
        check_correctness(128, 256)
        print()

        run_comprehensive_batch_norm_tests()