# Стоит ли тело головы MTP переводить на наш int8-путь.
# Формы -- НАСТОЯЩИЕ, из слепка, и разрезаны как при TP=2. Сравниваем ровно то, что
# делает сейчас движок (плотный bf16 matmul при M=1) с нашим w8a16_gemv.
import json, glob, os, sys, time, torch
sys.path.insert(0, "/mnt/d1/alex/VLLM_fa2/solutions/fa2_sm70_cutlass_grade")
import importlib.util
def w8a16_ext():
    # Компилятора в этой среде нет -- берём ГОТОВЫЙ .so из сборки стенда (тот же, что в бою).
    so = "/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_stend8095/fa2sm70_w8a16/fa2sm70_w8a16.so"
    сп = importlib.util.spec_from_file_location("fa2sm70_w8a16", so)
    м = importlib.util.module_from_spec(сп); сп.loader.exec_module(м)
    return м

MD = "/mnt/d1/alex/VLLM_ch/Qwen3.8-27B-ABL_int8_i12"
from safetensors import safe_open
idx = json.load(open(glob.glob(os.path.join(MD, "*.safetensors.index.json"))[0]))["weight_map"]

# (имя, ось разреза TP): 0 = по выходу (column-parallel), 1 = по входу (row-parallel)
СЛОИ = [
    ("mtp.fc.weight", 0),
    ("mtp.layers.0.self_attn.q_proj.weight", 0),
    ("mtp.layers.0.self_attn.k_proj.weight", 0),
    ("mtp.layers.0.self_attn.v_proj.weight", 0),
    ("mtp.layers.0.self_attn.o_proj.weight", 1),
    ("mtp.layers.0.mlp.gate_proj.weight", 0),
    ("mtp.layers.0.mlp.up_proj.weight", 0),
    ("mtp.layers.0.mlp.down_proj.weight", 1),
]
G = 128

def упаковать(W):  # W [K,N] fp16 -> qweight[K/4,N] int32, scales[K/G,N], qzeros
    K, N = W.shape
    Wg = W.reshape(K // G, G, N).float()
    мин = Wg.amin(1); макс = Wg.amax(1)
    амп = torch.maximum(макс.abs(), мин.abs()).clamp_min(1e-8)
    sc = (амп / 127.0)
    q = torch.clamp(torch.round(Wg / sc.unsqueeze(1)) + 128, 0, 255).to(torch.int32)
    q = q.reshape(K, N)
    qw = torch.zeros(K // 4, N, dtype=torch.int32, device=W.device)
    for i in range(4):
        qw |= (q[i::4] & 0xFF) << (8 * i)
    # gptq v1: хранится z-1, ядро прибавляет 1 обратно
    qz = torch.full((K // G, N // 4), 0, dtype=torch.int32, device=W.device)
    z = 127
    for i in range(4):
        qz |= (z & 0xFF) << (8 * i)
    return qw.contiguous(), qz.contiguous(), sc.half().contiguous()

def сек(f, повт=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(повт): f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / повт * 1e6  # мкс

ext = w8a16_ext()
всего_п = всего_н = байт_п = байт_н = 0.0
print(f"{'слой':44s} {'K':>6s} {'N':>6s} {'плотн мкс':>10s} {'наш мкс':>9s} {'x':>5s} {'отн.ошибка':>11s}")
for имя, ось in СЛОИ:
    with safe_open(os.path.join(MD, idx[имя]), framework="pt", device="cpu") as f:
        W = f.get_tensor(имя)
    # torch хранит [out,in]; режем как TP=2
    if ось == 0: W = W[: W.shape[0] // 2]
    else:        W = W[:, : W.shape[1] // 2]
    Wt = W.t().contiguous().half().cuda()       # [K,N]
    K, N = Wt.shape
    if K % (4 * G) or N % 4:
        print(f"{имя:44s} пропуск (K={K} N={N})"); continue
    x = torch.randn(1, K, dtype=torch.half, device="cuda")
    Wd = Wt.contiguous()
    qw, qz, sc = упаковать(Wt)
    y_п = x @ Wd
    y_н = ext.w8a16_gemv(x, qw, qz, sc, G, False)
    ош = ((y_н.float() - y_п.float()).norm() / y_п.float().norm()).item()
    t_п = сек(lambda: x @ Wd)
    t_н = сек(lambda: ext.w8a16_gemv(x, qw, qz, sc, G, False))
    всего_п += t_п; всего_н += t_н
    байт_п += K * N * 2; байт_н += K * N
    print(f"{имя:44s} {K:6d} {N:6d} {t_п:10.1f} {t_н:9.1f} {t_п/t_н:5.2f} {ош:11.2e}")
    del Wd, qw, qz, sc, Wt; torch.cuda.empty_cache()
print(f"\nПРОХОД (сумма): плотный {всего_п:.0f} мкс ({байт_п/2**20:.0f} МиБ), "
      f"наш {всего_н:.0f} мкс ({байт_н/2**20:.0f} МиБ), выигрыш x{всего_п/всего_н:.2f}")
print(f"ТРИ прохода за шаг: {3*всего_п/1000:.2f} мс -> {3*всего_н/1000:.2f} мс "
      f"(экономия {3*(всего_п-всего_н)/1000:.2f} мс)")
print(f"Освобождается памяти на ранг: {(байт_п-байт_н)/2**20:.0f} МиБ")
