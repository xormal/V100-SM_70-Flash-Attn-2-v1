#!/bin/bash
# Тот же честный прибор, но отбрасывает окна разгона (G<35) -- они завышали среднее.
LOG="$1"; MARK=$(wc -l < "$LOG")
M=huihui-qwen3.5-35b-a3b-abliterated-q8_0
curl -s -m 600 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Напиши подробный текст про историю паровых машин, 2000 слов.\"}],\"max_tokens\":2000,\"temperature\":0.0,\"ignore_eos\":true}" > /dev/null
sleep 12
tail -n +$MARK "$LOG" | grep -E "Avg generation throughput|SpecDecoding metrics" \
 | sed -E 's/.*Avg generation throughput: ([0-9.]+).*Running: ([0-9]+).*/G \1 \2/; s/.*Mean acceptance length: ([0-9.]+), Accepted throughput: ([0-9.]+).*/A \1 \2/' \
 | LC_ALL=C awk -v tag="$2" '
   $1=="G" {g=$2; run=$3; next}
   $1=="A" && g>35 && run>0 {a=$3; s=g-a; n++; ss+=1000/s; st+=g/s; g=0}
   END{if(n) printf "  %-8s окон=%d  шаг=%.2f мс  tau=%.3f\n", tag, n, ss/n, st/n; else print "  "tag" НЕТ ГОДНЫХ ОКОН"}'
