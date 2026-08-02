# -*- coding: utf-8 -*-
"""N-Omniglot 分卷断点续传下载器（figshare -> S3 预签名，10 秒过期，逐段刷新）。

用法: python download_nomniglot.py [file_id] [输出路径]
每次调用最多跑 --budget 秒，中断后再跑一次即可续传。
"""
import os
import subprocess
import sys
import urllib.request

FILES = {
    "bg1": ("31104472", "dvs_background_1.rar", 3013908031),
    "bg2": ("31104475", "dvs_background_2.rar", 2553351543),
    "eval": ("31104481", "dvs_evaluation.rar", 4239667693),
}


def fresh_url(file_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **kw):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(
        f"https://ndownloader.figshare.com/files/{file_id}",
        method="HEAD", headers={"User-Agent": "curl/8.0"})
    try:
        opener.open(req, timeout=20)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers["Location"]
        raise
    raise RuntimeError("no redirect")


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "bg2"
    budget = sys.argv[2] if len(sys.argv) > 2 else "280"
    file_id, name, total = FILES[key]
    out = os.path.join("data", "nomniglot_raw", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"{name}: 已有 {have/1e6:.0f}MB / {total/1e6:.0f}MB")
    if have >= total:
        print("已完整，无需下载")
        return
    url = fresh_url(file_id)
    rc = subprocess.call([
        "curl", "-sS", "-C", "-", "-o", out,
        "--max-time", budget, url])
    have = os.path.getsize(out) if os.path.exists(out) else 0
    print(f"本轮结束(rc={rc}): {have/1e6:.0f}MB / {total/1e6:.0f}MB "
          f"({100*have/total:.1f}%)")


if __name__ == "__main__":
    main()
