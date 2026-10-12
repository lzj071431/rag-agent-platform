# merge_txt.py
# -*- coding: utf-8 -*-
"""
合并两份清洗后的财报文本为一个知识库文件。

修复点（原版问题）：
1. 原版只是 f.write(content1); f.write(content2)，两份文档首尾直接粘连，
   中间没有任何分隔标记 —— 切分时会出现"跨公司"的混合 chunk，
   且 chunk 里丢失了"这段属于哪家公司"的信息，导致问"妙可蓝多的行权价格"
   时，正文块里根本没有"妙可蓝多"字样，向量检索匹配不上。
2. 新版为每份文档加上显式的分节标题（=====...=====），
   既给切分器提供了天然边界，也让 splitter.py 能给每个 chunk 打上
   "【来源】<公司名 + 文档名>" 的上下文前缀。
3. 合并过程完全无损：原始正文一字不改，只新增标题行，并做长度校验。
"""
import os
import re

# ---- 两份源文件 -> 对外展示的文档标题（会作为上下文前缀写入每个 chunk）----
SOURCES = [
    ("finance_clean.txt",
     "上海妙可蓝多食品科技股份有限公司《2025年股票期权激励计划（修订稿）》独立财务顾问报告"),
    ("yili_clean.txt",
     "内蒙古伊利实业集团股份有限公司 年度报告"),
]

OUTPUT = "competitor_data.txt"
MARKER = "====="


def write_section(f, title: str, body: str):
    f.write(f"{MARKER} 文件：{title} {MARKER}\n")
    f.write(body.rstrip("\n"))
    f.write("\n\n")


def main():
    total_src = 0
    sections = []

    for filename, title in SOURCES:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"缺少源文件：{filename}（请先用 preprocess_pdf.py 清洗）")
        with open(filename, "r", encoding="utf-8") as fp:
            body = fp.read()
        total_src += len(body)
        sections.append((filename, title, body))
        print(f"[读取] {filename:22s} {len(body):>9,} 字符 <- {title}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for _, title, body in sections:
            write_section(f, title, body)

    # ---- 无损校验：去掉新加的标题行后，必须与源文件逐字节一致 ----
    merged = open(OUTPUT, encoding="utf-8").read()
    stripped = re.sub(rf"^{MARKER} .*? {MARKER}\n", "", merged, flags=re.M)
    origin = "".join(b.rstrip("\n") + "\n\n" for _, _, b in sections)
    ok = stripped.strip() == origin.strip()
    # 逐份校验（更严格：确认每份正文都原样出现在合并文件里）
    per_ok = all(b.rstrip("\n") in merged for _, _, b in sections)

    print("-" * 66)
    print(f"[输出] {OUTPUT}")
    print(f"       源文件字符合计 : {total_src:,}")
    print(f"       合并后字符数   : {len(merged):,}（含 {(len(merged) - total_src)} 个新增标题字符）")
    print(f"       无损校验       : {'通过 ✅' if ok and per_ok else '失败 ❌'}")
    print(f"       分节数量       : {merged.count(MARKER + ' 文件：')} 个")
    if not (ok and per_ok):
        raise SystemExit("合并结果与源文件不一致，已中止！")
    print("✅ 合并完成！")


if __name__ == "__main__":
    main()
