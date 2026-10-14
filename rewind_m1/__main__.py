"""학습·평가·재개 CLI. 기본 데이터는 코드 검사용 synthetic fixture."""

import argparse
import json
import hashlib
import os
import tempfile
from pathlib import Path

from .training import Trainer, read_checkpoint, save_checkpoint


def save_report(report, path):
    """Checkpoint 다음에 같은 디렉터리의 완전한 JSON으로 교체한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(json.dumps(report, indent=2)+"\n")
        except Exception:
            f.close()
            temporary.unlink()
            raise
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description="M1 regression 기반: synthetic 또는 검증된 특징 계약")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=80, help="재개 포함 총 update 수")
    train.add_argument("--resume", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train","validation","test"), default="validation")
    evaluate.add_argument("--final-test", action="store_true")
    evaluate.add_argument("--output", type=Path, help="검증 provenance 포함 JSON 저장(선택)")
    for p in (train,evaluate):
        p.add_argument("--dataset", type=Path, help="검증 계약을 포함한 m1_features_v1 JSON; 생략 시 synthetic")
        p.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.dataset.read_text(encoding="utf-8")) if args.dataset else None
        # 동일 last.pt 재개 시 파일을 덮기 전에 원본 식별자를 고정한다.
        resume_hash = hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.command=="train" and args.resume else None
        checkpoint = read_checkpoint(args.resume if args.command=="train" else args.checkpoint) if args.command=="evaluate" or args.resume else None
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.command=="train" else checkpoint["config"]
        trainer = Trainer(config,document,args.device)
        if args.command=="evaluate":
            if checkpoint["manifest_hash"] != trainer.manifest_hash: raise ValueError("평가 dataset manifest 불일치")
            if checkpoint["source"]["code_sha256"] != trainer.source["code_sha256"]: raise ValueError("평가 code hash 불일치")
            trainer.model.load_state_dict(checkpoint["model"])
            result={"metrics":trainer.evaluate(args.split,args.final_test),"checkpoint_step":checkpoint["step"],
                "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),"manifest_hash":trainer.manifest_hash,
                "source":trainer.source,"environment":trainer.environment,"runtime_observation":trainer.runtime_observation,
                "input_contract":trainer.document["contract"]}
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(result,indent=2))
            return 0
        if args.steps < 1: raise ValueError("steps는 양의 정수 필요")
        if not args.resume and (args.output/"last.pt").exists(): raise ValueError("기존 checkpoint가 있습니다. 새 output 또는 --resume 사용")
        if checkpoint: trainer.resume(checkpoint)
        if args.steps < trainer.step: raise ValueError("총 steps가 재개 위치보다 작습니다.")
        start_step=trainer.step
        prior=None
        report_path=args.output/"report.json"
        if checkpoint and report_path.exists():
            prior=json.loads(report_path.read_text(encoding="utf-8"))
            if prior["step"]>start_step or prior["config"]!=config or prior["manifest_hash"]!=trainer.manifest_hash or prior["source"]["code_sha256"]!=trainer.source["code_sha256"]:
                raise ValueError("기존 보고서와 resume 위치 불일치. 분기 재개에는 새 output 사용")
        initial = trainer.evaluate("train")
        log = []
        recovery = []
        if prior and prior["step"] < start_step:
            recovery.append({"initial_step":prior["step"],"final_step":start_step,"report_recovery":True,
                "missing_log_steps":{"after":prior["step"],"through":start_step},
                "reason":"checkpoint 저장 후 report 저장 전 중단 또는 이전 report 복원; 누락 loss를 만들지 않음",
                "resume_checkpoint_sha256":resume_hash,"log":[]})

        def persist():
            final_train=trainer.evaluate("train")
            segment={"initial_step":start_step,"final_step":trainer.step,"resumed_from":str(args.resume) if args.resume else None,
                "resume_checkpoint_sha256":resume_hash,"initial_train":initial,"final_train":final_train,"log":list(log)}
            report={"step":trainer.step,"config":trainer.config,"manifest_hash":trainer.manifest_hash,
                "input_contract":trainer.document["contract"],"source":trainer.source,"environment":trainer.environment,
                "runtime_observation":trainer.runtime_observation,"initial_train":prior["initial_train"] if prior else initial,
                "initial_step":prior["initial_step"] if prior else start_step,"final_train":final_train,"validation":trainer.evaluate(),
                "log":(prior["log"] if prior else [])+log,"segments":(prior["segments"] if prior else [])+recovery+[segment]}
            save_checkpoint(trainer.checkpoint(),args.output/"last.pt")
            save_report(report, report_path)
            return report

        while trainer.step < args.steps:
            loss = trainer.train_step()
            if trainer.step%10==0 or trainer.step==args.steps:
                log.append({"step":trainer.step,"train_batch_loss":loss,"validation":trainer.evaluate()})
                persist()
        # updates가 없을 때도 resume endpoint를 보존한다.
        report=persist()
        print(json.dumps({"step":trainer.step,"initial_loss":initial["normalized_mse"],
            "final_loss":report["final_train"]["normalized_mse"],"data_kind":trainer.contract.data_kind,"output":str(args.output)},indent=2))
    except (ValueError,OSError,KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__=="__main__": raise SystemExit(main())
