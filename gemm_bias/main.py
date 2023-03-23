import numpy as np
import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    C = te.placeholder((M, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


# "llvm -mcpu=core-avx2"
# "llvm -mcpu=skylake-avx512"

target = tvm.target.Target("llvm")
#  [8192, 256] * [256, 128]; [8192, 128] * [128, 128]

M = 8192
K = 256
N = 128
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)


log_file = "matmul_add.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=100,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))


func = tvm.build(sch, args, target)
a_np = np.random.uniform(size=(M, K)).astype(np.float32)
b_np = np.random.uniform(size=(K, N)).astype(np.float32)
c_np = np.random.uniform(size=(M, N)).astype(np.float32)
out_np = a_np.dot(b_np) + c_np

dev = tvm.cpu()
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
c_tvm = tvm.nd.array(c_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(a_tvm, b_tvm, c_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)

print("Equivalent python schedule:")
print(task.print_best(log_file))
