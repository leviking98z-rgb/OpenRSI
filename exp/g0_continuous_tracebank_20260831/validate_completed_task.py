#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--require-steps',type=int,default=16)
    a=ap.parse_args()
    m=json.loads(a.manifest.read_text())
    stat_path=a.output_dir/'stat.json'; events_path=a.output_dir/'search_events.jsonl'; state_path=a.output_dir/'search_state.json'
    errors=[]
    for p in (stat_path,events_path,state_path):
        if not p.is_file(): errors.append(f'missing:{p.name}')
    stat=json.loads(stat_path.read_text()) if stat_path.is_file() else {}
    steps=[x for x in stat.get('steps',[]) if isinstance(x,dict)]
    ids=[int(x['step']) for x in steps if x.get('step') is not None]
    if len(steps)!=a.require_steps: errors.append(f'step_count:{len(steps)}')
    if sorted(ids)!=list(range(a.require_steps)): errors.append(f'non_contiguous_steps:{sorted(ids)}')
    by_step={int(x['step']):x for x in steps if x.get('step') is not None}
    node_by_step={i:str(x.get('node_id') or '') for i,x in by_step.items()}
    required=('operator','status','score','prompt_tokens','completion_tokens','model_time_used','sandbox_time_used','parent_steps','node_id')
    for i,x in sorted(by_step.items()):
        for k in required:
            if k not in x: errors.append(f'step_{i}_missing:{k}')
        for parent in x.get('parent_steps') or []:
            try: parent=int(parent)
            except Exception: errors.append(f'step_{i}_bad_parent:{parent!r}'); continue
            if parent not in by_step: errors.append(f'step_{i}_missing_parent:{parent}')
            elif parent>=i: errors.append(f'step_{i}_nonprior_parent:{parent}')
    events=[]
    if events_path.is_file():
        for n,line in enumerate(events_path.read_text().splitlines(),1):
            if not line.strip(): continue
            try: events.append(json.loads(line))
            except Exception as e: errors.append(f'event_{n}_json:{e}')
    generated={int(e['search_step']):e for e in events if e.get('event_type')=='node_generated' and e.get('search_step') is not None}
    evaluated={int(e['search_step']):e for e in events if e.get('event_type')=='node_evaluated' and e.get('search_step') is not None}
    if sorted(generated)!=list(range(a.require_steps)): errors.append(f'generated_steps:{sorted(generated)}')
    if sorted(evaluated)!=list(range(a.require_steps)): errors.append(f'evaluated_steps:{sorted(evaluated)}')
    seen=set()
    for i in range(a.require_steps):
        e=generated.get(i)
        if not e: continue
        pid=str(e.get('program_id') or '')
        if node_by_step.get(i) and pid!=node_by_step[i]: errors.append(f'step_{i}_node_event_mismatch')
        for parent in e.get('parent_ids') or []:
            if parent not in seen: errors.append(f'step_{i}_unknown_parent_id:{parent}')
        seen.add(pid)
    archive=Path(str(m.get('archive') or ''))
    if not archive.is_file(): errors.append('archive_missing')
    elif digest(archive)!=m.get('archive_sha256'): errors.append('archive_sha256_mismatch')
    if not m.get('success'): errors.append('manifest_success_false')
    if m.get('sample_index')!=0: errors.append(f'sample_index:{m.get("sample_index")}')
    if m.get('max_operator_executions')!=a.require_steps: errors.append(f'manifest_budget:{m.get("max_operator_executions")}')
    result={
      'ok':not errors,'task_id':m.get('task_id'),'worker_id':m.get('worker_id'),'slot':m.get('slot'),
      'steps':len(steps),'operators':{op:sum(str(x.get('operator') or x.get('mode')).lower()==op for x in steps) for op in ('draft','debug','improve','crossover')},
      'prompt_tokens':sum(int(x.get('prompt_tokens') or 0) for x in steps),
      'completion_tokens':sum(int(x.get('completion_tokens') or 0) for x in steps),
      'scored_steps':sum(x.get('score') is not None for x in steps),
      'archive':str(archive),'archive_sha256':m.get('archive_sha256'),'errors':errors,
    }
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True))
    raise SystemExit(0 if not errors else 1)
if __name__=='__main__': main()
