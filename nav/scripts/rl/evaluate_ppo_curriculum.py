"""Validation-first monitoring of immutable PPO checkpoints on fixed tasks."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from nav.scripts.rl.audit_pointgoal_ppo_result import EXPECTED_KEYS, audit_result, validate_report
from nav.scripts.rl.evaluate_ppo_checkpoints import write_json


def score(report):
    episodes=report["episodes"]
    collisions=sum(e["collision_steps"] for e in episodes)
    forwards=sum(e["executed_action_counts"].get("forward",0) for e in episodes)
    return dict(successes=sum(e["success"] for e in episodes), episodes=len(episodes),
        success_rate=sum(e["success"] for e in episodes)/len(episodes),
        collision_steps=collisions, forward_steps=forwards,
        collision_per_forward=collisions/max(1,forwards),
        rotation_steps=sum(e["executed_action_counts"].get("turn left",0)+e["executed_action_counts"].get("turn right",0) for e in episodes),
        steps=sum(e["steps_taken"] for e in episodes),
        warning_steps=sum(e["warning_steps"] for e in episodes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--initial-checkpoint",type=Path,required=True)
    parser.add_argument("--initial-benchmark-summary",type=Path,required=True)
    parser.add_argument("--initial-benchmark-checkpoint-sha256",required=True)
    parser.add_argument("--input-points",type=Path,required=True)
    parser.add_argument("--validation-manifest",type=Path,required=True)
    parser.add_argument("--unity",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--base-port",type=int,default=52000)
    parser.add_argument("--milestones",type=int,nargs="+",default=[25,50,100,200,300,400,500])
    parser.add_argument("--benchmark-milestones",type=int,nargs="+",default=[100,200,400,500])
    parser.add_argument("--poll-seconds",type=float,default=30)
    parser.add_argument("--pin-initial-budgets",action="store_true",
                        help="Pin canonical budgets to initial benchmark and later validation budgets to initial validation.")
    args=parser.parse_args()
    if not args.milestones or min(args.milestones)<=0 or not 1<=args.poll_seconds<=60:
        raise SystemExit("Positive milestones and a 1–60s poll interval are required")
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(args.initial_checkpoint)!=args.initial_benchmark_checkpoint_sha256:
        raise SystemExit("Initialization hash does not match the supplied benchmark evidence")
    initial_benchmark=json.loads(args.initial_benchmark_summary.read_text())
    validate_report(initial_benchmark)
    # Canonical benchmark bytes are fixed across all September 12/13 runs.
    if sha(args.input_points)!="1cb2062f53a68832ec2815bd62a61d0100c5fdcb58882602dda44f05127face2":
        raise SystemExit("The fixed input_points file changed")
    rows=[dict(json.loads(line),split="test") for line in args.validation_manifest.read_text().splitlines() if line.strip()]
    keys={(r["scene_name"],r["episode_id"]) for r in rows}
    counts=Counter(r["scene_name"] for r in rows)
    if len(keys)!=len(rows) or set(counts)!={f"scene{i}" for i in range(1,25)} or len(set(counts.values()))!=1:
        raise SystemExit("Validation needs unique, equally represented tasks from all 24 scenes")
    per_scene=next(iter(counts.values()))
    args.output_root.mkdir(parents=True,exist_ok=True)
    status_path=args.output_root/"monitor.json"
    identity=dict(initial_checkpoint_sha256=sha(args.initial_checkpoint),
        input_points_sha256=sha(args.input_points),validation_manifest_sha256=sha(args.validation_manifest))
    if args.pin_initial_budgets:
        identity.update(pin_initial_budgets=True,
                        initial_benchmark_summary_sha256=sha(args.initial_benchmark_summary))
    state=json.loads(status_path.read_text()) if status_path.exists() else dict(identity=identity,evaluations=[])
    if state["identity"]!=identity:
        raise SystemExit("Refusing to merge a different checkpoint or task set into this monitor")
    validation=args.output_root/"validation_tasks.jsonl"
    validation.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    common=[sys.executable,"-m","nav.scripts.evaluation.evaluate_pointgoal_policy",
        "--baseline","ppo","--unity",str(args.unity),"--device","cuda","--scenes","24",
        "--eval-seed","0","--workers",str(args.workers),"--base-port",str(args.base_port),
        "--persistent-policy","--resume","--safety-shield","--shield-collision-streak","8",
        "--shield-recovery-turns","6","--shield-recovery-forward-steps","4",
        "--shield-max-clearance-turns","8","--shield-warning-streak","8",
        "--shield-warning-turns","2","--shield-turn-streak","12","--shield-turn-escape-steps","4"]

    def evaluate(checkpoint,update,kind):
        directory=args.output_root/f"update_{update:06d}"/kind
        directory.mkdir(parents=True,exist_ok=True)
        summary=directory/"summary.json"
        source=(["--manifest",str(validation),"--episodes-per-scene",str(per_scene)]
                if kind=="validation" else ["--input-points",str(args.input_points),"--episodes-per-scene","4"])
        command=common+source+["--checkpoint",str(checkpoint),"--output-root",str(directory)]
        if args.pin_initial_budgets:
            reference = (args.initial_benchmark_summary if kind=="benchmark96" else
                         Path(state["evaluations"][0]["validation"]["summary"]) if state["evaluations"] else None)
            if reference is not None:
                command += ["--step-budget-reference",str(reference),
                            "--step-budget-reference-sha256",sha(reference)]
        command_path=directory/"command.json"
        if command_path.exists() and json.loads(command_path.read_text())!=command:
            raise RuntimeError("Evaluation command changed in an existing output directory")
        write_json(command_path,command)
        state.update(state="evaluating",current_update=update,current_split=kind)
        write_json(status_path,state)
        if not summary.exists():
            with (directory/"evaluation.log").open("a") as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode or not summary.exists():
                raise RuntimeError(f"Evaluation failed: {directory}")
        report=json.loads(summary.read_text())
        validate_report(report,expected_keys=keys if kind=="validation" else EXPECTED_KEYS)
        if kind=="benchmark96":
            write_json(directory/"protocol_audit.json",audit_result(report,initial_benchmark))
        elif state["evaluations"]:
            prior=json.loads(Path(state["evaluations"][0]["validation"]["summary"]).read_text())
            old={(e["scene_name"],e["episode_id"]):e["step_budget"] for e in prior["episodes"]}
            if any(e["step_budget"]!=old[(e["scene_name"],e["episode_id"])] for e in report["episodes"]):
                raise RuntimeError("Validation budgets differ from initialization")
        return dict(score(report),summary=str(summary))

    while True:
        evaluated={e["update"] for e in state["evaluations"]}
        available=[u for u in sorted(set(args.milestones)) if u not in evaluated and (args.checkpoint_dir/f"update_{u:06d}.pt").exists()]
        if 0 not in evaluated:
            update,checkpoint=0,args.initial_checkpoint
        elif available:
            update=available[0]
            checkpoint=args.checkpoint_dir/f"update_{update:06d}.pt"
        elif set(args.milestones)<=evaluated:
            state["state"]="completed"
            write_json(status_path,state)
            return
        else:
            state["state"]="waiting_for_checkpoint"
            write_json(status_path,state)
            time.sleep(args.poll_seconds)
            continue
        entry=dict(update=update,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        entry["validation"]=evaluate(checkpoint,update,"validation")
        previous_best=max((e["validation"]["successes"] for e in state["evaluations"]),default=-1)
        if update==0:
            entry["benchmark96"]=dict(score(initial_benchmark),summary=str(args.initial_benchmark_summary),reused=True)
        elif entry["validation"]["successes"]>=previous_best or update in args.benchmark_milestones:
            entry["benchmark96"]=evaluate(checkpoint,update,"benchmark96")
        state["evaluations"].append(entry)
        baseline=state["evaluations"][0]
        entry["validation_nonregression"]=entry["validation"]["successes"]>=baseline["validation"]["successes"]
        entry["validation_safety_gate"]=entry["validation"]["collision_per_forward"]<=max(
            baseline["validation"]["collision_per_forward"]*1.1,baseline["validation"]["collision_per_forward"]+.015)
        benchmark=entry.get("benchmark96",{})
        entry["benchmark_safety_gate"]=bool(benchmark) and benchmark["collision_per_forward"]<=max(
            baseline["benchmark96"]["collision_per_forward"]*1.1,baseline["benchmark96"]["collision_per_forward"]+.015)
        state["state"]="checkpoint_evaluated"
        state["target_candidate"]=bool(update>0 and benchmark.get("successes",0)>=77
            and entry["validation_nonregression"] and entry["validation_safety_gate"] and entry["benchmark_safety_gate"])
        write_json(status_path,state)
        print(json.dumps(entry),flush=True)
        if state["target_candidate"]:
            write_json(args.output_root/"target_candidate.json",entry)
            # This is a candidate, not a completed goal: repeat evaluation,
            # full validation, provenance, and safety audits are still needed.
            return


if __name__=="__main__":
    main()
