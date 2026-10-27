"""
final_20_run.py
===============
최종 실험 오케스트레이터.

두 가지 모드가 있다.

(1) 드라이버 모드  :  python final_20_run.py --all
    전체 실험을 순서대로, 각각 별도 자식 프로세스로 실행한다.
    한 스테이지가 죽어도 나머지는 계속된다. 이미 끝난 스테이지는 건너뛴다.

(2) 워커 모드      :  python final_20_run.py --stage S2 --outdir ... [--register-boiler]
    leakfix 패치를 설치한 뒤 run_all_xjtu_fixed.py 를 그 프로세스 안에서 실행한다.
    (드라이버가 내부적으로 호출한다. 직접 써도 된다.)

왜 자식 프로세스로 나누는가
    torch/OpenMP 상태를 스테이지마다 초기화하고, 한 스테이지의 크래시가
    전체를 죽이지 않게 하기 위해서다. PowerShell 이라 nohup/& 를 못 쓰므로
    파이썬이 직접 순차 실행한다.

실행 예
    set PYTHONUTF8=1
    python final_20_run.py --plan          (무엇을 돌릴지만 출력)
    python final_20_run.py --all           (전체 실행)
    python final_20_run.py --all --skip S4 S5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # code/ (스크립트 위치)
sys.path.insert(0, str(ROOT))
import paths                                     # noqa: E402  경로·seed 규약
RUNNER = ROOT / "run_all_xjtu_fixed.py"
BASE_OUT = paths.JOBS                            # work/jobs_out/final

BOILER_EXTRA = ["Boiler_1to3", "Boiler_2to1", "Boiler_3to2"]

PAIRS_20 = [
    "Sim2Real", "XB1toXB2", "XB2toXB1", "PB1toPB2", "PB2toPB1",
    "XB1toPB1", "XB2toPB2", "PB1toXB1", "PB2toXB2",
    "Boiler_1to2", "Boiler_2to3", "Boiler_3to1",
    "SMD_1-1to1-2", "SMD_1-2to1-1", "SMD_2-1to2-2", "SMD_3-1to3-2",
    "MSL_P-11toP-10", "MSL_P-10toP-11", "MSL_G-6toE-13", "MSL_G-6toE-1",
]


# ======================================================================
# 계획 정의
# ======================================================================
def build_plan(args) -> list[dict]:
    """(이름, 러너 인자, 출력 폴더) 목록."""
    plan = [
        {
            "name": "S2_main",
            "why": "오염 강건성. make_pool 이 반드시 불리므로 마스크가 여기서 전부 만들어진다. 가장 먼저.",
            "stages": ["S2"],
            "pairs": PAIRS_20,
            "seeds": paths.SEEDS_HEAVY,
            "outdir": BASE_OUT / "S2_main",
            "register_boiler": False,
        },
        {
            "name": "S1_main",
            "why": "주 결과. 마스크는 S2 에서 만들어진 캐시를 재사용.",
            "stages": ["S1"],
            "pairs": PAIRS_20,
            "seeds": paths.SEEDS_MAIN,
            "outdir": BASE_OUT / "S1_main",
            "register_boiler": False,
        },
        {
            "name": "BOILER6_S1",
            "why": "민감 그룹 n=3 -> n=6 확대. BENCH_PAIRS 런타임 등록 필요.",
            "stages": ["S1"],
            "pairs": BOILER_EXTRA,
            "seeds": paths.SEEDS_MAIN,
            "outdir": BASE_OUT / "BOILER6_S1",
            "register_boiler": True,
        },
        {
            "name": "BOILER6_S2",
            "why": "확장 3 pair 의 오염 곡선.",
            "stages": ["S2"],
            "pairs": BOILER_EXTRA,
            "seeds": paths.SEEDS_HEAVY,
            "outdir": BASE_OUT / "BOILER6_S2",
            "register_boiler": True,
        },
        {
            "name": "S4_alpha",
            "why": "MMD alpha 곡선. 누수 보정 지표로도 결론이 유지되는지 확인.",
            "stages": ["S4"],
            "pairs": PAIRS_20,
            "seeds": paths.SEEDS_HEAVY,
            "outdir": BASE_OUT / "S4_alpha",
            "register_boiler": False,
            # stage_alpha 는 pairs[:args.alpha_pairs] 로 자른다 (기본 8)
            "extra": ["--alpha_pairs", "20"],
        },
        {
            "name": "S5_baseline",
            "why": "TranAD / USAD 외부 baseline. 같은 평가 집합에서 비교해야 공정.",
            "stages": ["S5"],
            "pairs": PAIRS_20,
            "seeds": paths.SEEDS_HEAVY,
            "outdir": BASE_OUT / "S5_baseline",
            "register_boiler": False,
        },
    ]
    LEAKED = ["Sim2Real", "XB1toXB2", "XB2toXB1", "PB1toPB2", "PB2toPB1",
              "XB1toPB1", "XB2toPB2", "PB1toXB1", "PB2toXB2",
              "Boiler_1to2", "Boiler_2to3", "Boiler_3to1"]
    if args.leaked_only:
        # pool 제거량이 0 인 pair(SMD/MSL)는 auroc_nopool == auroc 이므로 재실행 불필요
        for p in plan:
            if p["name"] in ("S2_main", "S1_main"):
                p["pairs"] = LEAKED
                p["outdir"] = Path(str(p["outdir"]) + "_leaked")
                p["why"] += "  (누수 pair 12개만 재실행)"
    if args.smoke:
        for p in plan:
            p["pairs"] = p["pairs"][:1]
            p["seeds"] = [0]
            p["outdir"] = Path(str(p["outdir"]) + "_smoke")
            p["why"] = "SMOKE: " + p["why"]
    if args.skip:
        plan = [p for p in plan if p["name"] not in args.skip]
    if args.only:
        plan = [p for p in plan if p["name"] in args.only]
    return plan


# ======================================================================
# Boiler 확장 pair 등록
# ======================================================================
def _fields(cfg):
    if isinstance(cfg, dict):
        return dict(cfg)
    d = getattr(cfg, "__dict__", None)
    if d:
        return dict(d)
    if hasattr(cfg, "_asdict"):
        return dict(cfg._asdict())
    return {}


def _rebuild(cfg, newfields):
    if isinstance(cfg, dict):
        out = dict(cfg)
        out.update(newfields)
        return out
    new = copy.deepcopy(cfg)
    for k, v in newfields.items():
        try:
            setattr(new, k, v)
        except Exception:
            import dataclasses

            if dataclasses.is_dataclass(cfg):
                return dataclasses.replace(cfg, **newfields)
            raise
    return new


def derive_boiler_cfgs(bench_pairs: dict, log=print):
    """
    기존 Boiler_1to2 / 2to3 / 3to1 세 항목을 비교해서
    '어느 필드가 src 를 따라가고 어느 필드가 tgt 를 따라가는지'를 데이터로 알아낸다.
    추측하지 않고, 세 항목에서 실제로 관측된 값만 재조합한다.
    """
    known = {}
    for k in bench_pairs:
        m = re.fullmatch(r"Boiler_(\d)to(\d)", str(k))
        if m:
            known[k] = (int(m.group(1)), int(m.group(2)))
    if len(known) < 3:
        raise RuntimeError(
            f"기존 Boiler pair 가 3개 미만입니다({sorted(known)}). 자동 유도 불가."
        )

    fields_by_pair = {k: _fields(bench_pairs[k]) for k in known}
    names = set()
    for f in fields_by_pair.values():
        names |= set(f.keys())

    src_map: dict[str, dict[int, object]] = {}
    tgt_map: dict[str, dict[int, object]] = {}
    const: dict[str, object] = {}
    ambiguous: list[str] = []

    for name in sorted(names):
        vals = {k: fields_by_pair[k].get(name, None) for k in known}
        distinct = {repr(v) for v in vals.values()}
        if len(distinct) == 1:
            const[name] = next(iter(vals.values()))
            continue

        as_src, as_tgt = True, True
        for k, (s, t) in known.items():
            v = str(vals[k])
            if not (str(s) in v and str(t) not in v):
                as_src = False
            if not (str(t) in v and str(s) not in v):
                as_tgt = False
        if as_src and not as_tgt:
            src_map[name] = {known[k][0]: vals[k] for k in known}
        elif as_tgt and not as_src:
            tgt_map[name] = {known[k][1]: vals[k] for k in known}
        else:
            ambiguous.append(name)

    log(f"  상수 필드     : {sorted(const)}")
    log(f"  src 연동 필드 : {sorted(src_map)}")
    log(f"  tgt 연동 필드 : {sorted(tgt_map)}")
    if ambiguous:
        raise RuntimeError(
            "다음 필드가 src/tgt 어느 쪽인지 판별되지 않았습니다: "
            f"{ambiguous}\n"
            "boiler_extra.json 에 3개 pair 설정을 직접 적어 주세요 "
            "(--boiler-spec 로 지정)."
        )
    if not src_map or not tgt_map:
        raise RuntimeError("src 또는 tgt 연동 필드를 찾지 못했습니다. 수동 지정 필요.")

    template_key = next(iter(known))
    out = {}
    for newkey in BOILER_EXTRA:
        s, t = re.fullmatch(r"Boiler_(\d)to(\d)", newkey).groups()
        s, t = int(s), int(t)
        newfields = {}
        for name, mp in src_map.items():
            if s not in mp:
                raise RuntimeError(f"필드 {name} 에 src={s} 값이 관측되지 않았습니다.")
            newfields[name] = mp[s]
        for name, mp in tgt_map.items():
            if t not in mp:
                raise RuntimeError(f"필드 {name} 에 tgt={t} 값이 관측되지 않았습니다.")
            newfields[name] = mp[t]
        out[newkey] = _rebuild(bench_pairs[template_key], newfields)
    return out


def register_boiler(spec_path: str | None, log=print) -> None:
    import exp1_benchmark as B

    bp = B.BENCH_PAIRS
    missing = [k for k in BOILER_EXTRA if k not in bp]
    if not missing:
        log("[boiler] 3방향이 이미 등록되어 있음. 건너뜀")
        return

    if spec_path:
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        for k, v in spec.items():
            bp[k] = v
        log(f"[boiler] 수동 spec 으로 등록: {sorted(spec)}")
    else:
        log("[boiler] 기존 3개 항목에서 자동 유도 시도")
        derived = derive_boiler_cfgs(bp, log=lambda m: log("[boiler]" + m))
        for k, v in derived.items():
            bp[k] = v
        # 유도 결과를 기록해 둔다(다음에 검토/수정 가능)
        try:
            dump = {k: (_fields(v) if not isinstance(v, dict) else v) for k, v in derived.items()}
            (paths.WORK / "boiler_extra_derived.json").write_text(
                json.dumps(dump, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass
        log(f"[boiler] 등록 완료: {sorted(derived)} (boiler_extra_derived.json 에 기록)")

    # 검증: prepare 가 실제로 도는지, 그리고 원본 pair 와 데이터가 다른지
    try:
        import exp9_unified as E9

        for k in BOILER_EXTRA:
            log(f"[boiler] 검증 prepare({k}) ...")
            D = E9.prepare(k, None) if _prepare_takes_two(E9.prepare) else E9.prepare(k)
            shapes = {kk: getattr(vv, "shape", None) for kk, vv in (D or {}).items()
                      if hasattr(vv, "shape")}
            log(f"[boiler]   OK  {shapes}")
    except TypeError:
        log("[boiler] prepare 검증은 러너 인자(args)가 필요해 생략합니다(본 실행에서 확인됨)")
    except Exception as e:
        log(f"[boiler] 경고: prepare 검증 실패: {e!r}")
        log("[boiler] 그대로 진행합니다. 실패하면 errors.log 에 남습니다.")


def override_sim_cols(order, log=print) -> None:
    """
    sim-to-real pair 의 source 채널 순서를 바꿔 채널 대응의 영향을 측정한다.

    SIM_COLS / REAL_COLS (pipe_columns.py, work/data/pipe/columns.json) 는
    코드는 두 리스트를 '위치 기준'으로 대응시킨다. 이 대응이 물리적으로 옳은지는
    코드만으로 알 수 없으므로, 순서를 바꿔 돌려보고 결론이 달라지는지 확인한다.

    order 예)
      "0,1,2,3,4"  기본(현행)
      "0,1,4,3,2"  3번째와 5번째 채널 교환
      "shuffle"    무작위 순열(통제군: 대응이 무의미하다면 성능이 유지될 것)
    """
    try:
        import exp9_unified as E9
    except Exception as e:
        log(f"[simcols] import 실패: {e!r}")
        return
    base = list(E9.SIM_COLS)
    if order == "shuffle":
        import random

        idx = list(range(len(base)))
        random.Random(0).shuffle(idx)
    else:
        idx = [int(i) for i in order.split(",")]
    if sorted(idx) != list(range(len(base))):
        log(f"[simcols] 잘못된 순열: {idx}. 무시합니다.")
        return
    E9.SIM_COLS = [base[i] for i in idx]
    log("[simcols] source 채널 순서 변경")
    for a, b in zip(E9.SIM_COLS, E9.REAL_COLS):
        log(f"[simcols]   {a}  ->  {b}")


def register_msl(outdir, msl_max: int = 4, log=print) -> None:
    """
    MSL 4 pair 는 BENCH_PAIRS 에 없다. 러너의 all_pairs() 안에서
    register_msl_pairs(picked) 로 런타임 등록되는데, main() 은
    `pairs = args.pairs or XJTU_FIXED_PAIRS` 만 쓰므로 --pairs 를 주면
    all_pairs() 가 호출되지 않아 KeyError 가 난다.
    -> 러너를 모듈로 import 해서 all_pairs() 를 직접 한 번 돌려 준다.
    """
    try:
        import importlib

        RA = importlib.import_module("run_all_xjtu_fixed")
    except Exception as e:
        log(f"[msl] 러너 import 실패: {e!r}")
        return
    try:
        import exp1_benchmark as B

        before = set(B.BENCH_PAIRS)
        if hasattr(RA, "configure_output"):
            RA.configure_output(str(outdir))
        for sub in ("ckpt", "scores", "figures"):
            (Path(outdir) / sub).mkdir(parents=True, exist_ok=True)

        class _Stub:
            pass

        stub = _Stub()
        stub.msl_max = msl_max
        RA.all_pairs(stub)
        added = sorted(set(B.BENCH_PAIRS) - before)
        if added:
            log(f"[msl] 등록됨: {added}")
        elif any(str(k).startswith("MSL") for k in B.BENCH_PAIRS):
            log("[msl] 이미 등록되어 있음")
        else:
            log("[msl] MSL 데이터를 찾지 못했습니다. MSL pair 는 이번 실행에서 제외됩니다.")
    except Exception as e:
        log(f"[msl] all_pairs() 실행 실패: {e!r}  (MSL pair 는 제외됨)")


def _prepare_takes_two(fn) -> bool:
    import inspect

    try:
        return len(inspect.signature(fn).parameters) >= 2
    except Exception:
        return True


# ======================================================================
# 워커 모드
# ======================================================================
def run_worker(args) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.no_leakfix:
        sys.path.insert(0, str(ROOT))
        import final_10_leakfix as LF

        LF.install(
            outdir=str(outdir),
            hash_round=args.hash_round,
            abort_after_masks=args.masks_only,
            dump_scores=not args.no_dump_scores,
        )
    else:
        LF = None
        print("[worker] leakfix 비활성화 (auroc_nopool 없이 진행)")

    if args.sim_cols:
        override_sim_cols(args.sim_cols)

    register_msl(outdir, msl_max=args.msl_max)

    if args.register_boiler:
        register_boiler(args.boiler_spec)

    if args.deterministic:
        try:
            import torch

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[worker] cuDNN 결정성 모드 활성화")
        except Exception as e:
            print(f"[worker] 결정성 설정 실패: {e!r}")

    argv = [str(RUNNER), "--stages", *args.stage, "--outdir", str(outdir)]
    if args.pairs:
        argv += ["--pairs", *args.pairs]
    if args.seeds:
        argv += ["--seeds", *[str(s) for s in args.seeds]]
    if getattr(args, "seeds_main", None):
        argv += ["--seeds_main", *[str(s) for s in args.seeds_main]]
    if args.extra:
        argv += args.extra

    print("[worker] 러너 실행:", " ".join(argv))
    old_argv = sys.argv
    sys.argv = argv
    rc = 0
    t0 = time.time()
    try:
        import runpy

        runpy.run_path(str(RUNNER), run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 2
    finally:
        sys.argv = old_argv
        if LF is not None:
            try:
                LF.dump_summary(outdir / "leakfix_summary.json")
            except Exception:
                pass
        print(f"[worker] 소요 {time.time()-t0:.0f}s, rc={rc}")
    return rc


# ======================================================================
# 드라이버 모드
# ======================================================================
def reuse_ckpt(src_dir: Path, dst_dir: Path, log=print) -> None:
    """
    exp9_unified.pretrain() 은 outdir/ckpt 에 backbone 을 캐시하고, 있으면 재학습하지
    않는다. outdir 이 바뀌면 재학습되고 GPU 비결정성 때문에 결과가 미세하게 달라진다
    (실측: S2 960셀 중 78셀, 최대 0.057). 기존 ckpt 를 복사해 그 차이를 없앤다.
    """
    src = Path(src_dir) / "ckpt"
    dst = Path(dst_dir) / "ckpt"
    if not src.exists():
        log(f"[ckpt] 원본 없음: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("*.pt"):
        t = dst / f.name
        if not t.exists():
            shutil.copy2(f, t)
            n += 1
    log(f"[ckpt] {src} -> {dst} : {n}개 복사 (backbone 재학습 방지)")


def stage_done(outdir: Path) -> bool:
    f = outdir / "results.csv"
    if not f.exists():
        return False
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh) > 1
    except Exception:
        return False


def run_driver(args) -> int:
    plan = build_plan(args)
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    log_path = BASE_OUT / "driver_log.txt"

    print("=" * 74)
    print("최종 실험 계획")
    print("=" * 74)
    for i, p in enumerate(plan, 1):
        status = "완료(건너뜀)" if (stage_done(Path(p["outdir"])) and not args.force) else "실행"
        print(f"{i}. {p['name']:14s} stages={p['stages']} pairs={len(p['pairs'])} "
              f"seeds={p['seeds']} -> {status}")
        print(f"   목적: {p['why']}")
        print(f"   출력: {p['outdir']}")
    print("=" * 74)
    if args.plan:
        print("--plan 이므로 실행하지 않습니다.")
        return 0

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    if args.kmp_workaround:
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    rcs = {}
    with open(log_path, "a", encoding="utf-8") as logf:
        for p in plan:
            outdir = Path(p["outdir"])
            if stage_done(outdir) and not args.force:
                print(f"[driver] {p['name']} 이미 완료 -> 건너뜀")
                rcs[p["name"]] = 0
                continue

            if args.reuse_ckpt_from:
                reuse_ckpt(Path(args.reuse_ckpt_from), outdir)

            cmd = [
                sys.executable, str(ROOT / "final_20_run.py"),
                "--stage", *p["stages"],
                "--pairs", *p["pairs"],
                "--seeds", *[str(s) for s in p["seeds"]],
                "--outdir", str(outdir),
            ]
            if p["register_boiler"]:
                cmd.append("--register-boiler")
                if args.boiler_spec:
                    cmd += ["--boiler-spec", args.boiler_spec]
            if args.no_leakfix:
                cmd.append("--no-leakfix")
            if args.deterministic:
                cmd.append("--deterministic")
            if args.no_dump_scores:
                cmd.append("--no-dump-scores")
            if p.get("extra"):
                cmd += ["--extra", *p["extra"]]
            if args.hash_round is not None:
                cmd += ["--hash-round", str(args.hash_round)]

            head = f"\n{'='*74}\n[driver] {p['name']} 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n{' '.join(cmd)}\n"
            print(head)
            logf.write(head)
            logf.flush()

            t0 = time.time()
            stage_log = BASE_OUT / f"{p['name']}.log"
            with open(stage_log, "w", encoding="utf-8", errors="replace") as sf:
                proc = subprocess.run(cmd, cwd=str(paths.WORK), env=env,
                                      stdout=sf, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            if proc.returncode != 0:
                try:
                    tail = stage_log.read_text(encoding="utf-8", errors="replace")[-2500:]
                    print("---- 실패 로그 마지막 부분 ----")
                    print(tail)
                    print("-------------------------------")
                except Exception:
                    pass
            rcs[p["name"]] = proc.returncode
            tail = f"[driver] {p['name']} 종료 rc={proc.returncode} 소요 {dt/60:.1f}분\n"
            print(tail)
            logf.write(tail)
            logf.flush()

            if proc.returncode != 0 and not args.keep_going:
                print("[driver] 실패했습니다. --keep-going 을 주면 나머지도 계속합니다.")
                break

    print("=" * 74)
    for k, v in rcs.items():
        print(f"  {k:14s} rc={v}")
    bad = [k for k, v in rcs.items() if v != 0]
    print("=" * 74)
    if bad:
        print("실패:", ", ".join(bad))
        print("errors.log 와 jobs_out/final/driver_log.txt 를 확인하세요.")
        return 1
    print("전부 완료. 다음: python final_30_analysis.py")
    return 0


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="최종 실험 오케스트레이터")
    # 드라이버
    ap.add_argument("--all", action="store_true", help="전체 계획 실행")
    ap.add_argument("--plan", action="store_true", help="계획만 출력")
    ap.add_argument("--only", nargs="*", default=None, help="이 이름의 스테이지만")
    ap.add_argument("--skip", nargs="*", default=None, help="이 이름의 스테이지 제외")
    ap.add_argument("--force", action="store_true", help="완료된 것도 다시 실행")
    ap.add_argument("--reuse-ckpt-from", default=None,
                    help="이 폴더의 ckpt/*.pt 를 새 outdir 로 복사(backbone 재학습 방지)")
    ap.add_argument("--leaked-only", action="store_true",
                    help="S1/S2 를 누수 pair 12개로 한정(제거량 0인 pair는 재실행 불필요)")
    ap.add_argument("--keep-going", action="store_true", help="실패해도 계속")
    ap.add_argument("--smoke", action="store_true",
                    help="스테이지마다 pair 1개 seed 1개로 먼저 검증(오류 조기 발견)")
    ap.add_argument("--kmp-workaround", action="store_true",
                    help="KMP_DUPLICATE_LIB_OK=TRUE 설정(환경이 바뀌므로 기본 꺼짐)")
    # 워커
    ap.add_argument("--stage", nargs="*", default=None, help="러너에 넘길 스테이지")
    ap.add_argument("--pairs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=None)
    ap.add_argument("--seeds-main", "--seeds_main", dest="seeds_main", nargs="*", default=None,
                    help="주 결과표(S1)용 seed. 러너의 --seeds_main 으로 전달된다")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--register-boiler", action="store_true")
    ap.add_argument("--boiler-spec", default=None, help="수동 pair 설정 JSON 경로")
    ap.add_argument("--no-leakfix", action="store_true")
    ap.add_argument("--deterministic", action="store_true",
                    help="cuDNN 결정성 모드(느려지지만 실행 간 재현성 확보)")
    ap.add_argument("--sim-cols", default=None,
                    help="sim-to-real 의 source 채널 순열. 예: 0,1,4,3,2 또는 shuffle")
    ap.add_argument("--msl-max", type=int, default=4,
                    help="자동 선정할 MSL pair 수 (러너 all_pairs 와 동일 의미)")
    ap.add_argument("--no-dump-scores", action="store_true",
                    help="per-window 점수 npz 덤프 비활성화(기본은 덤프함)")
    ap.add_argument("--masks-only", action="store_true",
                    help="마스크만 만들고 각 pair 를 즉시 중단(빠른 점검용, errors.log 에 기록 남음)")
    ap.add_argument("--hash-round", type=int, default=None,
                    help="window 해시 비교 전 반올림 자릿수(기본: 바이트 정확 비교)")
    ap.add_argument("--extra", nargs="*", default=None, help="러너에 그대로 전달할 추가 인자")
    args, unknown = ap.parse_known_args()
    # argparse 는 nargs="*" 인 --extra 뒤에 오는 '-' 로 시작하는 인자를 받지 못한다.
    # 따라서 인식하지 못한 인자는 러너에게 그대로 넘긴다.
    if unknown:
        args.extra = (args.extra or []) + unknown
        print(f"[worker] 러너로 전달할 추가 인자: {unknown}")

    if not RUNNER.exists():
        print(f"[오류] 러너가 없습니다: {RUNNER}")
        return 1

    if args.stage:
        if not args.outdir:
            print("[오류] --stage 사용 시 --outdir 필수")
            return 1
        return run_worker(args)

    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
