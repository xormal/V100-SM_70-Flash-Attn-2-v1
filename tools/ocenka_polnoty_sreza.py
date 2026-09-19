# ПОЛНОТА КОРОТКОГО СПИСКА ПО СЖАТОМУ ПРЕДСТАВЛЕНИЮ.
# Словарь черновика читается 1212 МиБ на ранг и стоит 4.97 мс (88 % крыши чтения) -- ускорить
# нечем, можно только читать МЕНЬШЕ БАЙТ. int4 по ВСЕМ измерениям это делает (303 МиБ), но не
# влезает. Здесь проверяется, годится ли int4 по СРЕЗУ измерений: тогда 61-127 МиБ.
# Ядро i4 берёт K во время исполнения -- новой сборки не нужно (компилятора в среде нет).
# Два сжатия: наивный срез первых r измерений и СЛУЧАЙНАЯ проекция (та же память, но
# сохраняет скалярные произведения по лемме Джонсона-Линденштрауса).
import json, glob, os, sys, torch, importlib.util
from safetensors import safe_open
MD="/mnt/d1/alex/VLLM_ch/Qwen3.8-27B-ABL_int8_i12"
idx=json.load(open(glob.glob(os.path.join(MD,"*.safetensors.index.json"))[0]))["weight_map"]
so="/mnt/d1/alex/VLLM_ch/run/.fa2sm70_build_stend8095/fa2sm70_i4/fa2sm70_i4.so"
сп=importlib.util.spec_from_file_location("fa2sm70_i4",so)
i4=importlib.util.module_from_spec(сп); сп.loader.exec_module(i4)

h=torch.load("/mnt/d1/alex/mtp_skrytye_1250213.pt").cuda().half()
print(f"скрытых состояний: {tuple(h.shape)}")
with safe_open(os.path.join(MD, idx["lm_head.weight"]), framework="pt", device="cpu") as f:
    W=f.get_tensor("lm_head.weight")
V=W.shape[0]//2                        # доля ранга 0 при TP=2
W=W[:V].half().cuda()
print(f"голова ранга: {tuple(W.shape)}  ({W.numel()*2/2**20:.0f} МиБ)")

# точный местный аргмакс -- эталон
эт=torch.empty(h.shape[0],dtype=torch.long,device="cuda")
for i in range(0,h.shape[0],64):
    эт[i:i+64]=(h[i:i+64].float()@W.float().t()).argmax(-1)

def полнота(Q,S,hp,имя,миб):
    # ядро принимает M от 1 до 4 (шаг черновика ровно такой) -- идём четвёрками
    попал={N:0 for N in (8,16,64,256)}
    n=hp.shape[0]
    for i in range(0,n,4):
        сч=i4.scores(hp[i:i+4].contiguous(),Q,S)
        for N in попал:
            топ=сч.topk(N,dim=-1).indices
            попал[N]+=(топ==эт[i:i+4].unsqueeze(1)).any(1).sum().item()
    стр=[f"top{N}={попал[N]/n*100:5.1f}%" for N in (8,16,64,256)]
    print(f"  {имя:34s} {миб:6.1f} МиБ  " + "  ".join(стр))

torch.manual_seed(0)
print("\nНАИВНЫЙ СРЕЗ первых r измерений:")
for r in (512,1024,2048,5120):
    Q,Sc=i4.pack(W[:,:r].contiguous())
    полнота(Q,Sc,h[:,:r],f"срез r={r}",Q.numel()/2**20)
    del Q,Sc; torch.cuda.empty_cache()

print("\nСЛУЧАЙНАЯ ПРОЕКЦИЯ в r измерений:")
for r in (512,1024,2048):
    R=(torch.randn(W.shape[1],r,device="cuda",dtype=torch.float16)/ (r**0.5))
    Wp=torch.empty(V,r,device="cuda",dtype=torch.float16)
    for i in range(0,V,8192):
        Wp[i:i+8192]=(W[i:i+8192].float()@R.float()).half()
    Q,Sc=i4.pack(Wp.contiguous())
    hp=(h.float()@R.float()).half()
    полнота(Q,Sc,hp,f"проекция r={r}",Q.numel()/2**20)
    del Q,Sc,Wp,R,hp; torch.cuda.empty_cache()
