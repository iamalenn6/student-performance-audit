"""Собирает и выполняет ноутбук из src/stage4_experiments.py (формат «percent»).

Запуск из корня репозитория:  python src/build_notebook.py
Результат: notebooks/04_validation_models_experiments.ipynb (с сохранёнными выводами и графиками).
Нужен только стандартный стек (numpy, pandas, matplotlib, scikit-learn, scipy); Jupyter не требуется.
Если Jupyter установлен, то можно просто открыть .ipynb и выполнить «Run All».
"""
import ast, base64, io, json, os, sys, traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stage4_experiments.py"
OUT = ROOT / "notebooks" / "04_validation_models_experiments.ipynb"


def parse_cells(text):
    cells, cur, kind = [], [], None
    for line in text.splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, cur))
            kind, cur = ("markdown" if "[markdown]" in line else "code"), []
        else:
            cur.append(line)
    cells.append((kind, cur))
    out = []
    for kind, lines in cells:
        while lines and not lines[0].strip(): lines = lines[1:]
        while lines and not lines[-1].strip(): lines = lines[:-1]
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        out.append((kind, "\n".join(lines)))
    return out


def lines_of(s):
    ls = s.split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]


def run_cell(src, ns, count):
    outputs = []
    buf = io.StringIO()

    def flush():
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": lines_of(buf.getvalue().rstrip("\n") + "\n")})
            buf.seek(0); buf.truncate(0)

    def display(obj):
        flush()
        data = {"text/plain": lines_of(repr(obj))}
        if hasattr(obj, "to_frame") and not hasattr(obj, "to_html"):
            obj = obj.to_frame()
        if hasattr(obj, "to_html"):
            data["text/plain"] = lines_of(obj.to_string())
            data["text/html"] = lines_of(obj.to_html())
        outputs.append({"output_type": "display_data", "data": data, "metadata": {}})

    ns["display"] = display
    tree = ast.parse(src)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    with redirect_stdout(buf):
        exec(compile(tree, "<cell>", "exec"), ns)
        val = eval(compile(last, "<cell>", "eval"), ns) if last is not None else None
    flush()
    if val is not None:
        outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {},
                        "data": {"text/plain": lines_of(repr(val))}})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
    plt.close("all")
    return outputs


def main():
    cells, nb_cells, ns, count = parse_cells(SRC.read_text(encoding="utf-8")), [], {"__name__": "__main__"}, 0
    os.chdir(ROOT / "notebooks")                       # как при запуске Jupyter из папки notebooks/
    for kind, src in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines_of(src)})
            continue
        count += 1
        try:
            outs = run_cell(src, ns, count)
        except Exception:
            traceback.print_exc(); sys.exit(f"Ошибка в ячейке {count}")
        nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": count, "source": lines_of(src), "outputs": outs})
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}}}
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Готово:", OUT.relative_to(ROOT), f"({len(nb_cells)} ячеек)")


if __name__ == "__main__":
    main()
