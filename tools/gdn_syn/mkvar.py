# Генератор вариантов для стенда синтеза: раскладывает ядра четырёх стратегий по ПОДПРОСТРАНСТВАМ
# ИМЁН (fa2_sm70::gdn::vsN), чтобы все они жили в ОДНОЙ единице трансляции и мерились вперемешку.
# ВАЖНО: сгенерированные vorig/vs1..vs4.cuh — архив четырёх стратегий.
# Для генерации укажите корень с вариантами через GDN_VARIANTS_ROOT; исходный снимок берётся
# из этого репозитория.
import re, sys, os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = os.environ.get("GDN_VARIANTS_ROOT")
if not W:
    raise SystemExit("set GDN_VARIANTS_ROOT to the external directory containing wf_dd5b381d-* variants")
srcs = {
  # ИСХОДНОЕ ядро -- ЗАМОРОЖЕННЫЙ СНИМОК, а не живой файл репозитория. Ловушка, стоившая
# одного прогона: генератор читал живой файл, тот уже был заменён отгружаемым ядром, и
# "опора" молча стала отгружаемым ядром с чужим (77 КБ) размером разделяемой -> 1 блок.
  "vorig": R + "/tools/gdn_syn/orig_snapshot.cuh",
  "vs1":   W + "/wf_dd5b381d-a6d-1/solutions/fa2_sm70_cutlass_grade/fa2_sm70/csrc/volta_gdn_o.cuh",
  "vs2":   W + "/wf_dd5b381d-a6d-2/solutions/fa2_sm70_cutlass_grade/fa2_sm70/csrc/volta_gdn_o.cuh",
  "vs3":   W + "/wf_dd5b381d-a6d-3/solutions/fa2_sm70_cutlass_grade/fa2_sm70/csrc/volta_gdn_o.cuh",
  "vs4":   W + "/wf_dd5b381d-a6d-4/solutions/fa2_sm70_cutlass_grade/fa2_sm70/csrc/volta_gdn_o.cuh",
}
out = R + "/tools/gdn_syn"
for tag, p in srcs.items():
    t = open(p).read()
    t = t.replace("#pragma once", "")
    t = re.sub(r'#include "volta_warp_mma.h"', "", t)
    t = re.sub(r'#include <cuda_fp16\.h>|#include <cuda_runtime\.h>|#include <math_constants\.h>', "", t)
    # sub-namespace
    assert "namespace gdn {" in t, tag
    t = t.replace("namespace gdn {", "namespace gdn {\nnamespace %s {" % tag, 1)
    i = t.rindex("}  // namespace gdn")
    t = t[:i] + "}  // namespace %s\n" % tag + t[i:]
    t = t.replace("accumulate_wideB", "accumulate")   # S3: primitive lives only in its worktree; WIDE_H measured negative
    open(os.path.join(out, tag + ".cuh"), "w").write("// СГЕНЕРИРОВАНО tools/gdn_syn/mkvar.py -- НЕ ПРАВИТЬ РУКАМИ\n" + t)
    print(tag, len(t))
