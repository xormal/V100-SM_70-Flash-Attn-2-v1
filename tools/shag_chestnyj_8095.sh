#!/bin/bash
# Честный прибор шага на стенде 8095. G и A из ОДНОГО окна журнала:
# шагов/с = G-A, шаг = 1/(G-A), tau = G/(G-A). Окна разгона (G<20) отбрасываются.
LOG="$1"; TAG="$2"; NTOK="${3:-1200}"; REQ="${4:-Напиши подробный текст про историю паровых машин, 2000 слов.}"
MARK=$(wc -l < "$LOG")
M=$(curl -s -m 10 http://127.0.0.1:8095/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -s -m 900 http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d "$(python3 -c "
import json,sys
print(json.dumps({'model':'$M','messages':[{'role':'user','content':'''$REQ'''}],
                  'max_tokens':$NTOK,'temperature':0.0,'ignore_eos':True}))")" > /dev/null
sleep 12
tail -n +$MARK "$LOG" | grep -E "Avg generation throughput|SpecDecoding metrics" \
 | sed -E 's/.*Avg generation throughput: ([0-9.]+).*Running: ([0-9]+).*/G \1 \2/; s/.*Mean acceptance length: ([0-9.]+), Accepted throughput: ([0-9.]+).*/A \1 \2/' \
 | LC_ALL=C awk -v tag="$TAG" '
   $1=="G" {g=$2; run=$3; next}
   $1=="A" && g>20 && run>0 {a=$3; s=g-a; n++; ss+=1000/s; st+=g/s; sg+=g; g=0}
   END{if(n) printf "  %-22s окон=%2d  шаг=%6.2f мс  tau=%.3f  ток/с=%5.1f\n", tag, n, ss/n, st/n, sg/n;
       else print "  "tag"  НЕТ ГОДНЫХ ОКОН"}'
