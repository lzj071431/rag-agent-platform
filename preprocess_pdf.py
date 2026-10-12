#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_pdf.py —— PDF → 干净 UTF-8 TXT 的预处理脚本（面向 RAG 知识库）

功能概览
--------
1. 用 pymupdf(fitz) 逐页提取文本；
2. 清洗：
   - 删除目录页行（连续点号引导线）与纯页码行；
   - 删除重复出现的页眉/页脚（可选）；
   - 修复“一个字一行”的碎片（把连续的碎片行拼回完整句子，遇到句末标点收尾）；
   - 合并连续空行、去除行首行尾空白、压缩多余空格、去掉中文字符之间的伪空格；
3. 质量检查：中文占比过低 / 出现乱码（U+FFFD、私用区 PUA）/ 几乎无文本时，
   打印醒目警告，并提示改用 OCR 或从 Word 原版复制；
4. 以 UTF-8（无 BOM）写入 <stem>_clean.txt；
5. 终端打印清洗前后的字符数对比与各项清洗统计。

用法
----
    python preprocess_pdf.py                          # 自动寻找 thesis.pdf / finance.pdf
    python preprocess_pdf.py thesis.pdf               # 指定输入 → thesis_clean.txt
    python preprocess_pdf.py finance.pdf --lang en    # 英文文档，关闭中文占比告警
    python preprocess_pdf.py thesis.pdf --lang zh --sort --strict
    python preprocess_pdf.py thesis.pdf -o out.txt --keep-page-numbers --no-margin-strip

依赖
----
    pip install pymupdf
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# pymupdf >= 1.24 的新包名；旧环境只有 fitz，两者都兼容
try:
    import pymupdf  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf  # type: ignore
    except ImportError:
        sys.exit("[错误] 未安装 PyMuPDF。请先执行：pip install pymupdf")


# ============================================================================
# 一、可调参数（按自己的 PDF 特点微调）
# ============================================================================

SHORT_LINE_MAX = 2          # 行长 <= 此值视为“一个字一行”的碎片，触发拼接
NORMAL_LINE_MIN = 12        # （宽松模式）拼接结果达到此长度即视为“正常句子”，可以收尾
MERGE_MAX_LEN = 80          # 拼接硬上限：防止把整页文字吞成一整行

# 只在“连续出现多个碎片行”时才判定为一个字一行的排版故障。
# 原因：中文论文里“摘要”“目录”“结论”“致谢”这类 2 字标题很常见，
# 若见短行就与下一行拼接，会把这些标题粘进正文；而真正的逐字排版必然是连续成片的。
# 想严格按“只要 1-2 字就与下一行拼接”执行，把此常量改为 False 即可。
REQUIRE_RUN_FOR_MERGE = True

TOC_DOT_MIN = 4             # 连续点号 >= 此值即判定为目录行（如 "........"）
PAGE_NUM_DROP = True        # 是否删除纯页码行（可用 --keep-page-numbers 关闭）

MARGINAL_MAX_LEN = 40       # 判定重复页眉/页脚时的最大行长
MARGINAL_RATIO = 0.5        # 同一行出现在 >= 50% 页面的页边位置 → 视为页眉/页脚
MARGINAL_EDGE_LINES = 2     # 只看每页前 N 行 / 后 N 行是否为页眉页脚

# 句子结束标点：拼接时遇到这些字符即认为一句话说完了
# （逗号、顿号属于“软标点”，不作为收尾依据，否则段落会被切得很碎）
PUNCT_END = "。！？；：!?;:.…"

# --- 预编译正则 -------------------------------------------------------------
# 目录行：4 个以上连续点号/中点，或 2 个以上省略号（兼容全角）
RE_TOC_DOTS = re.compile(
    r"(?:[.．]\s*){%d,}|(?:[·・]\s*){%d,}|(?:。\s*){%d,}|(?:…\s*){2,}"
    % (TOC_DOT_MIN, TOC_DOT_MIN, TOC_DOT_MIN)
)

# 纯页码行：12 / -12- / [12] / (12) / 第12页 / Page 12 / 12/100 / 共12页
RE_PAGE_NUMBER = re.compile(
    r"""^\s*(?:
          [-–—\[(【]?\s*\d{1,4}\s*[-–—\])】]?      # 12、-12-、[12]、(12)、【12】
        | 第\s*\d{1,4}\s*页                        # 第 12 页
        | 页\s*[:：]?\s*\d{1,4}
        | page\s*[:：]?\s*\d{1,4}                  # Page 12
        | \d{1,4}\s*/\s*\d{1,4}                    # 12/100
        | 共\s*\d{1,4}\s*页
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")       # 控制字符
RE_MULTISPACE = re.compile(r"[ \t\u3000]{2,}")                   # 连续空格/全角空格
RE_CJK_SPACE = re.compile(r"(?<=[\u3400-\u9fff])[ \t]+(?=[\u3400-\u9fff])")  # 汉字间伪空格
RE_MULTI_BLANK = re.compile(r"\n{3,}")                           # 连续空行
RE_MEANINGFUL = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")          # 有意义的字符

# 统计用正则
RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
RE_LATIN = re.compile(r"[A-Za-z]")
# 乱码特征：替换符 U+FFFD、私用区 PUA（加密/自定义字体的典型产物）、控制字符
RE_GARBAGE = re.compile(r"[\ufffd\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ============================================================================
# 二、文本提取
# ============================================================================

def extract_pages(pdf_path: Path, sort: bool = False) -> tuple[list[str], int]:
    """逐页提取文本。

    返回 (每页文本列表, 无文本页数)。无文本页通常意味着该页是扫描图片。
    """
    page_texts: list[str] = []
    empty_pages = 0

    with pymupdf.open(str(pdf_path)) as doc:
        if doc.needs_pass:
            raise RuntimeError("PDF 已加密，需要密码才能提取文本。")
        for page in doc:
            # "text" 模式按人类阅读顺序输出行；sort=True 按坐标排序，
            # 对多栏排版 / 表格更有用，但偶尔会打乱顺序
            try:
                txt = page.get_text("text", sort=sort)
            except TypeError:  # 极旧版本的 PyMuPDF 不支持 sort 关键字
                txt = page.get_text("text")
            if not txt or not txt.strip():
                empty_pages += 1
            page_texts.append(txt)

    return page_texts, empty_pages


def strip_repeated_marginal_lines(
    page_texts: list[str], ratio: float = MARGINAL_RATIO, max_len: int = MARGINAL_MAX_LEN
) -> tuple[list[str], int]:
    """删除在多数页面页边位置重复出现的行（页眉、页脚、水印文字）。

    只在每页的前 MARGINAL_EDGE_LINES 行 / 后 MARGINAL_EDGE_LINES 行里找，
    这样正文中间恰好重复出现的句子不会被误删。
    """
    if len(page_texts) < 3:  # 页数太少，谈不上“重复”
        return page_texts, 0

    counter: Counter[str] = Counter()
    for txt in page_texts:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            continue
        edge = lines[:MARGINAL_EDGE_LINES] + lines[-MARGINAL_EDGE_LINES:]
        counter.update({ln for ln in edge if len(ln) <= max_len})

    threshold = max(2, int(len(page_texts) * ratio))
    noisy = {ln for ln, n in counter.items() if n >= threshold}
    if not noisy:
        return page_texts, 0

    removed = 0
    result: list[str] = []
    for txt in page_texts:
        lines = txt.splitlines()
        keep: list[str] = []
        for i, ln in enumerate(lines):
            is_edge = i < MARGINAL_EDGE_LINES or i >= len(lines) - MARGINAL_EDGE_LINES
            if is_edge and ln.strip() in noisy:
                removed += 1
                continue
            keep.append(ln)
        result.append("\n".join(keep))

    return result, removed


# ============================================================================
# 三、逐行清洗与过滤
# ============================================================================

def is_toc_line(line: str) -> bool:
    """目录行判定：包含大量连续点号（引导线），或整行大部分是点号。"""
    if RE_TOC_DOTS.search(line):
        return True
    dots = line.count(".") + line.count("．") + line.count("…")
    return dots >= TOC_DOT_MIN and dots / max(len(line), 1) > 0.3


def is_page_number_line(line: str) -> bool:
    """纯页码行判定，如 '12'、'- 12 -'、'第 12 页'、'Page 12'、'12/100'。"""
    return bool(RE_PAGE_NUMBER.match(line))


def is_noise_line(line: str) -> bool:
    """无意义行判定：不含任何汉字/字母/数字（例如纯符号线 '———'、'◆◇◆'）。

    例外：短小的句末标点行（如孤立的 '。'）不算噪声 —— 逐字排版的 PDF 里，
    句号常常单独占一行，必须留给拼接逻辑贴回上一句，否则整段会丢掉句末标点。
    """
    if RE_MEANINGFUL.search(line):
        return False
    if len(line) <= 4 and any(ch in PUNCT_END or ch in "，、,）】" for ch in line):
        return False
    return True


def normalize_line(line: str) -> str:
    """行内规整：去控制字符、去首尾空白、压缩空格、删除汉字之间的伪空格。"""
    line = RE_CTRL.sub("", line)
    line = line.strip()
    line = RE_MULTISPACE.sub(" ", line)
    # PDF 提取常把每个汉字之间塞上空格，这里把它们连回去（不影响英文单词间空格）
    line = RE_CJK_SPACE.sub("", line)
    return line.strip()


def _smart_join(left: str, right: str) -> str:
    """拼接两行：英文/数字之间补一个空格，中文之间直接相连；顺带修复英文断词连字符。"""
    if left.endswith("-") and right[:1].islower():
        return left[:-1] + right           # 英文换行断词："informa-" + "tion"
    if left[-1:].isascii() and right[:1].isascii() and left[-1:].isalnum() and right[:1].isalnum():
        return f"{left} {right}"
    return left + right


def _punct_only(line: str) -> bool:
    """整行是否只有标点（无汉字/字母/数字）。"""
    return bool(line) and RE_MEANINGFUL.search(line) is None


def merge_broken_lines(lines: list[str], stats: dict) -> list[str]:
    """修复“一个字一行”：把连续的碎片行拼接成完整句子。

    规则：
      - 遇到长度 <= SHORT_LINE_MAX 的行，且它后面紧跟的也是碎片行 → 认定进入了
        “一个字一行”的故障区段，开始累积拼接（REQUIRE_RUN_FOR_MERGE 控制）；
      - 累积到句末标点（。！？…）就收尾；为防止整页被吞成一行，另设
        MERGE_MAX_LEN 硬上限；
      - 碎片区段结束（遇到正常长度的行或空行）时立即收尾；
      - 孤立的纯标点行（如单独一行的“。”）直接贴回上一行末尾。
    """
    out: list[str] = []
    n = len(lines)

    def is_frag(s: str) -> bool:
        return 0 < len(s) <= SHORT_LINE_MAX

    def next_content(idx: int) -> str:
        while idx < n and not lines[idx]:
            idx += 1
        return lines[idx] if idx < n else ""

    i = 0
    while i < n:
        line = lines[i]

        # 空行：保留为段落边界，最后统一压缩
        if not line:
            out.append("")
            i += 1
            continue

        # --- 情况 1：进入“一个字一行”的碎片区段，开始拼接 ---
        if is_frag(line) and (not REQUIRE_RUN_FOR_MERGE or is_frag(next_content(i + 1))):
            buf = line
            j = i + 1
            while j < n and lines[j]:
                nxt = lines[j]
                # 宽松模式会吞掉正常长度的下一行；严格模式（默认）只吃碎片
                if REQUIRE_RUN_FOR_MERGE and not is_frag(nxt):
                    break
                buf = _smart_join(buf, nxt)
                j += 1
                # 严格模式：拼到句末标点为止，绝不按长度截断（避免把句子切两半）
                limit = MERGE_MAX_LEN if REQUIRE_RUN_FOR_MERGE else NORMAL_LINE_MIN
                if buf[-1] in PUNCT_END or len(buf) >= limit:
                    break
            out.append(buf)
            stats["merge_groups"] += 1
            i = j
            continue

        # --- 情况 2：孤立的纯标点短行，贴回上一行 ---
        if is_frag(line) and _punct_only(line) and out and out[-1]:
            out[-1] += line
            i += 1
            continue

        # --- 情况 3：正常行，原样保留 ---
        out.append(line)
        i += 1

    return out


def clean_text(
    page_texts: list[str],
    drop_page_numbers: bool = PAGE_NUM_DROP,
    strip_margins: bool = True,
) -> tuple[str, dict]:
    """完整清洗流程，返回 (清洗后文本, 统计信息)。"""
    stats: dict = {
        "marginal_lines": 0,
        "toc_lines": 0,
        "pagenum_lines": 0,
        "noise_lines": 0,
        "merge_groups": 0,
        "blank_collapsed": 0,
    }

    # 1) 先按“整页”维度删掉重复页眉页脚
    if strip_margins:
        page_texts, stats["marginal_lines"] = strip_repeated_marginal_lines(page_texts)

    raw_text = "\n".join(page_texts)

    # 2) 逐行归一化 + 过滤目录行 / 页码行 / 噪声行
    kept: list[str] = []
    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            kept.append("")
            continue
        if is_toc_line(line):
            stats["toc_lines"] += 1
            continue
        if drop_page_numbers and is_page_number_line(line):
            stats["pagenum_lines"] += 1
            continue
        if is_noise_line(line):
            stats["noise_lines"] += 1
            continue
        kept.append(line)

    # 3) 修复“一个字一行”
    merged = merge_broken_lines(kept, stats)

    # 4) 压缩连续空行（保留一个空行作为段落分隔，对 RAG 分块友好）
    text = "\n".join(merged)
    stats["blank_collapsed"] = len(RE_MULTI_BLANK.findall(text))
    text = RE_MULTI_BLANK.sub("\n\n", text)
    text = text.strip() + "\n" if text.strip() else ""

    stats["clean_lines"] = len(text.splitlines())
    stats["clean_chars"] = len(text)
    return text, stats


# ============================================================================
# 四、提取质量检查（乱码 / 扫描版 / 中文占比过低）
# ============================================================================

def analyze_quality(text: str, lang: str, min_cjk_ratio: float) -> dict:
    """统计字符构成，判断提取质量；返回统计字典（含 warnings 列表）。"""
    visible = [ch for ch in text if not ch.isspace()]
    total = len(visible)
    cjk = len(RE_CJK.findall(text))
    latin = len(RE_LATIN.findall(text))
    garbage = len(RE_GARBAGE.findall(text))

    cjk_ratio = cjk / total if total else 0.0
    latin_ratio = latin / total if total else 0.0
    garbage_ratio = garbage / total if total else 0.0

    # 期望语言：auto 模式下先用“提取结果”粗略推断，再据此决定告警口径
    if lang == "auto":
        if cjk >= 20:
            expected = "zh"
        elif latin_ratio >= 0.5:
            expected = "en"
        else:
            expected = "unknown"
    else:
        expected = lang

    warnings: list[str] = []

    if total < 50:
        warnings.append(
            "几乎提取不到有效文本（可见字符 %d 个）。\n"
            "    → 该 PDF 很可能是【扫描件/图片型 PDF】，PyMuPDF 只能拿到图片，拿不到文字。\n"
            "    → 建议：用 OCR 处理，例如 `ocrmypdf -l chi_sim in.pdf out.pdf`，\n"
            "      或 PaddleOCR / ABBYY FineReader；也可以从 Word 原版直接复制文字。" % total
        )
    if garbage_ratio > 0.02:
        warnings.append(
            "检测到大量乱码字符（U+FFFD 替换符 / 私用区 PUA，占比 %.1f%%）。\n"
            "    → 常见原因：PDF 使用了【加密字体或自定义编码字体】，文本层无法直接解码。\n"
            "    → 建议：改用 OCR 重新识别，或从 Word 原版（.docx）复制正文内容。" % (garbage_ratio * 100)
        )
    if expected == "zh" and cjk_ratio < min_cjk_ratio:
        warnings.append(
            "中文字符占比极低（%.1f%%，阈值 %.1f%%），疑似乱码或字体编码异常。\n"
            "    → 建议：改用 OCR（ocrmypdf / PaddleOCR / ABBYY）或从 Word 原版复制。"
            % (cjk_ratio * 100, min_cjk_ratio * 100)
        )
    if expected == "unknown":
        warnings.append(
            "无法判断文档语言（中文占比 %.1f%%，英文字母占比 %.1f%%），文本疑似为乱码。\n"
            "    → 若这是中文论文，请加参数 --lang zh 复核；若确实提取异常，请改用 OCR。"
            % (cjk_ratio * 100, latin_ratio * 100)
        )

    return {
        "visible_chars": total,
        "cjk": cjk,
        "cjk_ratio": cjk_ratio,
        "latin": latin,
        "latin_ratio": latin_ratio,
        "garbage": garbage,
        "garbage_ratio": garbage_ratio,
        "expected": expected,
        "warnings": warnings,
    }


# ============================================================================
# 五、输入输出与命令行
# ============================================================================

def resolve_input(pdf_arg: str | None) -> Path:
    """确定输入 PDF：优先命令行参数，其次 thesis.pdf / finance.pdf，最后目录下唯一的 PDF。"""
    if pdf_arg:
        p = Path(pdf_arg).expanduser()
        if not p.is_file():
            sys.exit(f"[错误] 找不到输入文件：{p}")
        return p

    for name in ("thesis.pdf", "finance.pdf"):
        p = Path.cwd() / name
        if p.is_file():
            return p

    pdfs = sorted(Path.cwd().glob("*.pdf"))
    if len(pdfs) == 1:
        return pdfs[0]
    if not pdfs:
        sys.exit("[错误] 当前目录没有找到 PDF。请把 thesis.pdf 放到项目根目录，"
                 "或运行：python preprocess_pdf.py 你的文件.pdf")
    sys.exit("[错误] 当前目录有多个 PDF：%s\n       请显式指定其中之一。"
             % ", ".join(p.name for p in pdfs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF → 干净 UTF-8 TXT（面向 RAG 知识库的数据预处理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", nargs="?", default=None,
                        help="输入 PDF；省略时自动寻找 ./thesis.pdf 或 ./finance.pdf")
    parser.add_argument("-o", "--output", default=None,
                        help="输出 TXT 路径；默认 <输入文件名>_clean.txt")
    parser.add_argument("--lang", choices=["auto", "zh", "en"], default="auto",
                        help="文档语言，决定质量告警口径；默认 auto（自动判断）")
    parser.add_argument("--min-cjk-ratio", type=float, default=0.20,
                        help="中文占比告警阈值，默认 0.20（即 20%%）")
    parser.add_argument("--sort", action="store_true",
                        help="按坐标排序提取文本（多栏排版/表格推荐）")
    parser.add_argument("--no-margin-strip", action="store_true",
                        help="不删除重复页眉/页脚")
    parser.add_argument("--keep-page-numbers", action="store_true",
                        help="保留纯页码行")
    parser.add_argument("--preview-chars", type=int, default=300,
                        help="终端预览的字符数，0 表示不预览；默认 300")
    parser.add_argument("--strict", action="store_true",
                        help="质量检查不通过时以退出码 2 结束（便于接入流水线）")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台编码可能是 GBK，避免打印特殊字符时崩溃
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass

    args = build_parser().parse_args(argv)

    pdf_path = resolve_input(args.pdf)
    out_path = (Path(args.output).expanduser() if args.output
                else pdf_path.with_name(f"{pdf_path.stem}_clean.txt"))

    print("=" * 68)
    print(f"输入文件 : {pdf_path.resolve()}")
    print(f"输出文件 : {out_path.resolve()}")
    print("=" * 68)

    # --- 提取 ---
    try:
        page_texts, empty_pages = extract_pages(pdf_path, sort=args.sort)
    except Exception as exc:  # 损坏文件、加密文件等
        sys.exit(f"[错误] 读取 PDF 失败：{exc}")

    raw_text = "\n".join(page_texts)
    raw_chars = len(raw_text)
    n_pages = len(page_texts)

    print(f"总页数   : {n_pages}"
          + (f"（其中 {empty_pages} 页没有可提取文本，可能是扫描图）" if empty_pages else ""))

    # --- 清洗 ---
    clean, stats = clean_text(
        page_texts,
        drop_page_numbers=not args.keep_page_numbers,
        strip_margins=not args.no_margin_strip,
    )

    # --- 写出（UTF-8，无 BOM，换行统一为 \n） ---
    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(clean)
    except OSError as exc:
        sys.exit(f"[错误] 写出文件失败：{exc}")

    # --- 质量检查 ---
    quality = analyze_quality(clean, args.lang, args.min_cjk_ratio)

    # --- 统计与预览 ---
    removed = raw_chars - len(clean)
    pct = (removed / raw_chars * 100) if raw_chars else 0.0
    print()
    print("清洗统计")
    print("-" * 68)
    print(f"  目录行删除        : {stats['toc_lines']} 行")
    print(f"  纯页码行删除      : {stats['pagenum_lines']} 行")
    print(f"  页眉/页脚删除     : {stats['marginal_lines']} 行")
    print(f"  纯符号噪声行删除  : {stats['noise_lines']} 行")
    print(f"  碎片拼接          : {stats['merge_groups']} 处（修复“一个字一行”）")
    print(f"  压缩连续空行      : {stats['blank_collapsed']} 处")
    print("-" * 68)
    print(f"  字符数（清洗前）  : {raw_chars:,}")
    print(f"  字符数（清洗后）  : {len(clean):,}")
    print(f"  净减少            : {removed:,}（{pct:.1f}%）")
    print(f"  行数（清洗后）    : {stats['clean_lines']:,}")
    print(f"  中文字符占比      : {quality['cjk_ratio'] * 100:.1f}%"
          f"（{quality['cjk']:,} / {quality['visible_chars']:,} 非空白字符）")
    print(f"  英文字母占比      : {quality['latin_ratio'] * 100:.1f}%")
    print(f"  疑似乱码字符      : {quality['garbage']:,}"
          f"（{quality['garbage_ratio'] * 100:.1f}%）")

    # --- 质量告警 ---
    if quality["warnings"]:
        print()
        print("!" * 68)
        print("⚠  提取质量警告：")
        print("!" * 68)
        for i, msg in enumerate(quality["warnings"], 1):
            print(f"  [{i}] {msg}")
        print("!" * 68)
        print("  说明：以上告警不影响文件写出，但正文可能不可用；")
        print("        请先人工检查预览内容，再决定是否清洗入库（RAG）。")
        print("!" * 68)
    else:
        print("\n✓ 提取质量正常（中文字符占比与乱码比例都在合理范围）。")

    # --- 预览 ---
    if args.preview_chars > 0 and clean:
        preview = clean[: args.preview_chars].replace("\n", "\n  ")
        print(f"\n正文预览（前 {args.preview_chars} 个字符）")
        print("-" * 68)
        print("  " + preview + ("..." if len(clean) > args.preview_chars else ""))

    print(f"\n✓ 已写出 UTF-8 文本：{out_path.resolve()}")

    if args.strict and quality["warnings"]:
        print("\n[strict 模式] 质量检查未通过，退出码 2。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
