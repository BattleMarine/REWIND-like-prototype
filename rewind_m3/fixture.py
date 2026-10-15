"""Colab/로컬 CPU에서 작은 named-body→hand 세 경로를 재현하는 진입점."""
import argparse
import json
from pathlib import Path

from rewind_m1.__main__ import save_report
from rewind_m2.__main__ import main as body_main
from .__main__ import main as hand_main
from .data import synthetic_document,body_document

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description='CPU synthetic M3: body 고정 + upper/none 별도 학습 + 세 경로 평가')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--body-steps',type=int,default=80)
    p.add_argument('--hand-steps',type=int,default=100)
    args = p.parse_args(argv)
    if args.body_steps<1 or args.hand_steps<1:
        p.error('steps는 양의 정수 필요')
    if args.output.exists():
        p.error('fixture에는 새 output 필요; 재개는 개별 train CLI 사용')
    args.output.mkdir(parents=True)
    document = synthetic_document()
    whole,body = args.output/'whole.json',args.output/'body.json'
    save_report(document,whole);save_report(body_document(document),body)
    body_out = args.output/'body'
    body_main(['train','--config',str(ROOT/'configs/m3_body_synthetic.json'),'--output',str(body_out),'--dataset',str(body),'--steps',str(args.body_steps),'--device','cpu'])
    checkpoint = body_out/'last.pt'
    for mode in ('upper','none'):
        out = args.output/('hand_'+mode)
        hand_main(['train','--config',str(ROOT/f'configs/m3_hand_{mode}_synthetic.json'),'--output',str(out),'--dataset',str(whole),
                   '--body-checkpoint',str(checkpoint),'--steps',str(args.hand_steps),'--device','cpu'])
        sources = ('gt_upper_diagnostic','predicted_frozen_body') if mode=='upper' else ('none',)
        for source in sources:
            hand_main(['evaluate','--checkpoint',str(out/'last.pt'),'--body-checkpoint',str(checkpoint),'--dataset',str(whole),
                       '--condition',source,'--output',str(args.output/(source+'.json')),'--device','cpu'])
    results = {source:json.loads((args.output/(source+'.json')).read_text(encoding='utf-8'))
               for source in ('none','gt_upper_diagnostic','predicted_frozen_body')}
    save_report({'evaluation_kind':'synthetic_code_check','body_checkpoint':str(checkpoint),
                 'comparison_policy':'none is separately trained; GT diagnostic/predicted share upper model; all attach predicted body wrists',
                 'results':results},args.output/'comparison.json')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
