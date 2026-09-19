"""ОФЛАЙНОВЫЙ ПРОХОД ГОЛОВЫ MTP -- основание для её обучения (единственная незакрытая ось tau).

Порядок работ ОБРАТНЫЙ привычному и это главное: сперва гейт, потом обучение. Четыре прежних
захода обучения черновика сгорели именно на приборе -- он не воспроизводил боевое число, и дни
уходили на оптимизацию не той функции. Поэтому здесь два гейта, и первый работает БЕЗ ЖЕЛЕЗА:

  ГЕЙТ 1 (процессор): ПОКРЫТИЕ ВЕСОВ. Каждый тензор `mtp.*` из слепка обязан быть прочитан
     проходом РОВНО один раз. Ловит самый частый класс ошибки воспроизведения -- забытый или
     дважды использованный вес (у DFlash2 это был «подавался выход fc за вход»).
  ГЕЙТ 2 (железо, когда вернётся): ТОЖДЕСТВО С ДВИЖКОМ. На тройках из
     `FA2SM70_MTP_GATE_DUMP` (признаки цели, входные токены, токен, выданный черновиком) этот
     проход обязан дать ТОТ ЖЕ токен. До его прохождения никакая разность между моделями не
     осмысленна.

Порядок прохода снят из vllm/model_executor/models/qwen3_next.py (класс Qwen3NextMTP и
Qwen3NextDecoderLayer), а не угадан:
    emb   = embed_tokens(x_{t+1})
    emb   = pre_fc_norm_embedding(emb);  h = pre_fc_norm_hidden(h_t)
    z     = fc(cat([emb, h], -1))                       # 2*5120 -> 5120
    слой:  residual = z; y = input_layernorm(z)
           q,k,v -> q_norm/k_norm по голове -> RoPE (частичный 0.25, theta 1e7, neox)
           внимание причинное, GQA 24/4 -> o_proj
           y, residual = post_attention_layernorm(y, residual)   # слитое: residual += y
           y = down(silu(gate(y)) * up(y));  residual += y
    h'    = norm(residual);  логиты = lm_head(h')
"""
import json, glob, os, sys, math
import torch
from safetensors import safe_open

MD = os.environ.get("FA2SM70_MODEL", "/mnt/d1/alex/VLLM_ch/Qwen3.8-27B-ABL_int8_i12")


def _конфиг():
    c = json.load(open(os.path.join(MD, "config.json")))
    return c.get("text_config", c)


def загрузить_голову(устр="cpu", тип=torch.float32):
    """Все тензоры mtp.* из слепка. Возвращает (веса, счётчик чтений)."""
    инд = json.load(open(glob.glob(os.path.join(MD, "*.safetensors.index.json"))[0]))["weight_map"]
    имена = sorted(k for k in инд if k.startswith("mtp."))
    веса, по_файлам = {}, {}
    for k in имена:
        по_файлам.setdefault(инд[k], []).append(k)
    for ф, кл in по_файлам.items():
        with safe_open(os.path.join(MD, ф), framework="pt", device="cpu") as f:
            for k in кл:
                веса[k] = f.get_tensor(k).to(тип).to(устр)
    return веса


class Голова:
    """Проход головы MTP. Каждое обращение к весу ПРОТОКОЛИРУЕТСЯ -- это и есть гейт 1."""

    def __init__(self, веса, cfg, устр="cpu", тип=torch.float32):
        self.в, self.cfg, self.устр, self.тип = веса, cfg, устр, тип
        self.прочитано = {}

    def _w(self, имя):
        self.прочитано[имя] = self.прочитано.get(имя, 0) + 1
        return self.в[имя]

    def _норма(self, x, имя, eps):
        w = self._w(имя)
        d = x.to(torch.float32)
        d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + eps)
        return (d * w.to(torch.float32)).to(x.dtype)

    def _норма_по_голове(self, x, имя, eps):
        """q_norm/k_norm считаются по ПОСЛЕДНЕЙ оси (head_dim), а не по скрытой."""
        return self._норма(x, имя, eps)

    def _роуп(self, q, k, поз):
        cfg = self.cfg
        hd = cfg["head_dim"]
        доля = cfg.get("partial_rotary_factor", 1.0)
        rot = int(hd * доля)
        rot -= rot % 2
        тета = float(cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 1e4)))
        инв = 1.0 / (тета ** (torch.arange(0, rot, 2, device=q.device, dtype=torch.float32) / rot))
        уг = поз.to(torch.float32).unsqueeze(-1) * инв           # [L, rot/2]
        cos, sin = уг.cos(), уг.sin()

        def пов(t):                                             # neox: пары (i, i+rot/2)
            ч = t[..., :rot]
            x1, x2 = ч[..., : rot // 2], ч[..., rot // 2: rot]
            c = cos.unsqueeze(1); s = sin.unsqueeze(1)           # [L,1,rot/2]
            return torch.cat([torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1),
                              t[..., rot:]], dim=-1)
        return пов(q), пов(k)

    def проход(self, h, токены, вложения, lm_head=None):
        """h: [L, 5120] признаки цели; токены: [L] токены x_{t+1}; вложения: [V,5120] (или callable)."""
        cfg = self.cfg
        eps = float(cfg["rms_norm_eps"])
        H = cfg["hidden_size"]; hd = cfg["head_dim"]
        nq = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
        L = h.shape[0]
        emb = вложения(токены) if callable(вложения) else вложения[токены]
        emb = self._норма(emb, "mtp.pre_fc_norm_embedding.weight", eps)
        hh = self._норма(h, "mtp.pre_fc_norm_hidden.weight", eps)
        z = torch.cat([emb, hh], dim=-1) @ self._w("mtp.fc.weight").t()
        П = "mtp.layers.0."
        residual = z
        y = self._норма(z, П + "input_layernorm.weight", eps)
        # ГЕЙТОВАННОЕ ВНИМАНИЕ Qwen3-Next (`attn_output_gate=True` по умолчанию).
        # q_proj даёт 12288 = nq * 2 * hd, и разрез идёт ПО ГОЛОВЕ: каждая голова несёт
        # [q 256 | гейт 256]. Половинить тензор целиком было бы НЕВЕРНО -- форма бы сошлась,
        # а смысл нет. Снято из qwen3_next.py (Qwen3NextAttention.forward), а не угадано.
        qg = (y @ self._w(П + "self_attn.q_proj.weight").t()).view(L, nq, 2 * hd)
        q, гейт = torch.chunk(qg, 2, dim=-1)                 # [L, nq, hd] каждый
        k = (y @ self._w(П + "self_attn.k_proj.weight").t()).view(L, nkv, hd)
        v = (y @ self._w(П + "self_attn.v_proj.weight").t()).view(L, nkv, hd)
        q = self._норма_по_голове(q.contiguous(), П + "self_attn.q_norm.weight", eps)
        k = self._норма_по_голове(k, П + "self_attn.k_norm.weight", eps)
        поз = torch.arange(L, device=h.device)
        q, k = self._роуп(q, k, поз)
        гр = nq // nkv
        k = k.repeat_interleave(гр, dim=1); v = v.repeat_interleave(гр, dim=1)
        ат = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(hd)
        маска = torch.full((L, L), float("-inf"), device=h.device).triu(1)
        ат = (ат + маска).softmax(-1)
        о = torch.einsum("hqk,khd->qhd", ат, v.float()).to(h.dtype)       # [L, nq, hd]
        о = о * torch.sigmoid(гейт)                                       # ГЕЙТ после внимания
        y = о.reshape(L, nq * hd) @ self._w(П + "self_attn.o_proj.weight").t()
        residual = residual + y
        y = self._норма(residual, П + "post_attention_layernorm.weight", eps)
        врт = torch.nn.functional.silu(y @ self._w(П + "mlp.gate_proj.weight").t()) \
            * (y @ self._w(П + "mlp.up_proj.weight").t())
        y = врт @ self._w(П + "mlp.down_proj.weight").t()
        residual = residual + y
        hсл = self._норма(residual, "mtp.norm.weight", eps)
        if lm_head is None:
            return hсл
        return hсл @ (lm_head.t() if not callable(lm_head) else lm_head)


def гейт_покрытия():
    cfg = _конфиг()
    веса = загрузить_голову()
    г = Голова(веса, cfg)
    H = cfg["hidden_size"]; L = 7
    h = torch.randn(L, H)
    ток = torch.randint(0, 1000, (L,))
    вл = torch.randn(1000, H)                       # подставка вместо словаря 248320x5120
    вых = г.проход(h, ток, вл)
    непрочит = sorted(set(веса) - set(г.прочитано))
    дважды = {k: v for k, v in г.прочитано.items() if v != 1}
    print(f"  тензоров mtp.* в слепке: {len(веса)}")
    print(f"  прочитано проходом: {len(г.прочитано)}")
    print(f"  форма выхода: {tuple(вых.shape)} (ожидается ({L}, {H}))")
    if непрочит:
        print("  НЕ ПРОЧИТАНЫ (дефект воспроизведения):")
        for k in непрочит: print("    ", k)
    if дважды:
        print("  прочитаны НЕ ОДИН раз:", дважды)
    ок = (not непрочит) and (not дважды) and tuple(вых.shape) == (L, H) and torch.isfinite(вых).all()
    print(f"  ГЕЙТ 1 (покрытие весов): {'ПРОЙДЕН' if ок else 'ПРОВАЛЕН'}")
    return ок


if __name__ == "__main__":
    sys.exit(0 if гейт_покрытия() else 1)
