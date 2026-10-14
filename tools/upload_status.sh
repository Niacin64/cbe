#!/bin/bash
# 一条命令查看索引上传进度
PID=$(ps -eo pid,cmd | awk '/[h]f_upload_indexes\.py/ {print $1; exit}')
echo "── 上传进程 ──"
if [ -n "$PID" ]; then
  rc=$(( $(awk '/rchar/{print $2}' /proc/$PID/io) / 1048576 ))
  echo "  运行中 (PID $PID)，本次会话已从磁盘读取 ${rc} MiB"
else
  echo "  未运行（用 bash /tmp/hf_run_upload.sh 可续传）"
fi
echo
echo "── 远端已提交的文件 ──"
unset ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1" no_proxy="localhost,127.0.0.1"
export HF_TOKEN="$(tr -d '\r\n' < /home/liqingchen/.hf_token_hf)"
export HF_HUB_DISABLE_PROGRESS_BARS=1
/home/liqingchen/micromamba/envs/grace/bin/python - <<'PY'
from huggingface_hub import HfApi
fs=[f for f in HfApi().list_repo_tree('Niacin64/cbe-indexes',repo_type='dataset',recursive=True) if hasattr(f,'size') and f.size and f.rfilename.endswith('.faiss')]
tot=sum(f.size for f in fs)
for f in sorted(fs, key=lambda x: x.rfilename):
    print(f"  {f.rfilename:28s} {f.size/2**30:7.3f} GiB")
print(f"  合计 {tot/2**30:.2f} / 17.49 GiB ({tot/2**30/17.49*100:.0f}%)")
PY
echo
echo "── 最近日志 ──"
grep -v "FutureWarning\|warnings.warn" /tmp/hf_upload.log 2>/dev/null | tail -4
