"""Build the standalone CPU-only finalist comparison notebook."""
import argparse
import json
from pathlib import Path
from scripts.v210_tail import canonical, sha
from scripts.v210_tail_build import cell

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "outputs/v2_plan_inputs"


def build(out):
    plan = {}
    for seed in (42, 71, 103, 137, 181):
        arms = {}
        for version, arm, name in ((211, "head", "body_head"), (212, "lora", "body_lora")):
            final = seed in (137, 181)
            run = f"v{version}_final_results_{20261122 if version == 211 else 20261123}" if final else f"v{version}_r2_t4_results_20261122"
            raw = (INPUTS / run / f"seed-{seed}-summary.json").read_bytes()
            doc = json.loads(raw)
            assert doc["measurement"]["seed"] == seed and doc["measurement"]["method"] == f"v2.1.{version%10}"
            arms[arm] = {"sha256": sha(raw), "schema": doc["schema"],
                         "folder": f"v{version}_{name}128_a_{'final_' if final else ''}v1"}
        plan[str(seed)] = arms
    modules = {name: (ROOT/"scripts"/name).read_text(encoding="utf-8") for name in
               ("v210_tail.py", "v211_paired.py", "v2_final_paired_cpu.py")}
    support = (ROOT/"scripts/v210_tail_colab.py").read_text(encoding="utf-8").split("def run_tail_notebook")[0]
    command = support + "\n" + f"MODULES = {modules!r}\nPLAN = {plan!r}\n" + RUN
    compile(command, "finalist-cpu", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name":"python3","display_name":"Python 3"}},
                "cells": [cell("markdown","guide","CPU 전용 · 같은 클립과 seed에서 출력층과 LoRA를 비교합니다. CPU 런타임에서 설정 셀을 켜고 실행하세요."),
                          cell("code","settings",'RUN_COMPARE = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"'),
                          cell("code","run",command)]}
    out.mkdir()
    path = out/"v2_final_paired_cpu.ipynb"
    path.write_bytes(canonical(notebook))
    (out/"receipt.json").write_bytes(canonical({"sha256":sha(path.read_bytes()), "inputs":plan}))


RUN = """
if RUN_COMPARE:
    from google.colab import drive
    started = time.monotonic()
    drive.mount('/content/drive', timeout_ms=120000)
    reports = (Path(DRIVE_PROJECT)/'reports').resolve(strict=True)
    if not reports.is_relative_to(Path('/content/drive/MyDrive').resolve(strict=True)):
        raise ValueError('Drive project path')
    dest = reports/'v2_final_paired_cpu'/('run-'+uuid.uuid4().hex)
    dest.mkdir(parents=True)
    result = {'status':'started', 'files':{}}
    publish_tail_json(dest/'started.json',result)
    try:
        with tempfile.TemporaryDirectory(dir='/content') as tmp:
            work = Path(tmp)
            if shutil.disk_usage(work).free < 128*1024**2:
                raise OSError('CPU scratch space')
            for name, source in MODULES.items():
                (work/name).write_text(source, encoding='utf-8')
            (work/'plan.json').write_text(json.dumps(PLAN))
            subprocess.run([sys.executable,str(work/'v2_final_paired_cpu.py'),str(reports),str(work/'plan.json'),str(work/'output')],
                           check=True,timeout=min(780,max(1,900-(time.monotonic()-started))),
                           env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'))
            for name in ('paired-clips.csv','paired-summary.json'):
                result['files'][name] = publish_tail_file(dest/name,small_tail_file(work/'output'/name,4*1024**2),4*1024**2)
            result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=str(exc)[:1500])
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        print(dest,publish_tail_json(dest/'result.json',result),result['status'])
"""
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    build(parser.parse_args().output)
