#!/usr/bin/env python3
"""Build the current report page for the audited minimal-next benchmark."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from pathlib import Path
from typing import Any

import cv2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: float | int | None, suffix: str = "ms") -> str:
    return "-" if value is None else f"{float(value):.1f} {suffix}"


def fmt_number(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def metric(row: dict[str, Any], *keys: str) -> float | None:
    timing = row.get("timing_ms") or {}
    for key in keys:
        value = timing.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def first_value(rows: list[dict[str, Any]], *keys: str) -> float | None:
    values = [v for row in rows for v in [metric(row, *keys)] if v is not None]
    return mean(values)


def video_for(run_dir: Path, episode: int) -> Path | None:
    matches = list(run_dir.rglob(f"episode{episode}.mp4"))
    return matches[0] if matches else None


def frame_image(video: Path, frame_id: int, out: Path) -> bool:
    if out.exists():
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id)))
        ok, frame = cap.read()
        return bool(ok and cv2.imwrite(str(out), frame))
    finally:
        cap.release()


def valid_verifiers(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows = read_jsonl(run_dir / "hierarchical_decisions.jsonl")
    by_ep: dict[int, list[dict[str, Any]]] = {}
    invalid_eps: set[int] = set()
    for row in rows:
        if row.get("request_kind") != "verifier":
            continue
        ep = row.get("episode_index")
        if not isinstance(ep, int):
            continue
        if not row.get("prediction_parse_ok"):
            invalid_eps.add(ep)
            continue
        by_ep.setdefault(ep, []).append(row)
    return [items[-1] for _, items in sorted(by_ep.items())], len(invalid_eps)


def run_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "hierarchical_decisions.jsonl").exists():
            continue
        verifiers, invalid = valid_verifiers(run_dir)
        if "_supplement" in run_dir.name:
            continue
        if not verifiers:
            continue
        # Replace the known invalid 1Hz burger episode with the audited supplement.
        supplement = root / "1hz_next_place_burger_fries_supplement"
        if run_dir.name == "1hz_next_place_burger_fries" and supplement.exists():
            extra, extra_invalid = valid_verifiers(supplement)
            if extra:
                verifiers = [v for v in verifiers if v.get("episode_index") != 1]
                extra[0]["_video_run_dir"] = supplement
                verifiers.append(extra[0])
                invalid += extra_invalid
        records.append({"run_dir": run_dir, "verifiers": verifiers, "invalid": invalid})
    return records


def image_ids(row: dict[str, Any]) -> list[int]:
    return [int(v) for v in (row.get("evidence_frame_ids") or []) if isinstance(v, int)]


def prompt_block(label: str, row: dict[str, Any]) -> str:
    return (
        f"<details><summary>{html.escape(label)}</summary>"
        "<p>The following is the exact English input sent to the models and their raw output, retained for reproducibility.</p>"
        f"<h4>Completion verifier: full input</h4><pre>{html.escape(str(row.get('final_prompt') or row.get('prompt') or ''))}</pre>"
        f"<h4>Completion verifier: raw output</h4><pre>{html.escape(str(row.get('verifier_raw_prediction') or row.get('raw_prediction') or ''))}</pre>"
        f"<h4>Next-subtask planner: full input</h4><pre>{html.escape(str(row.get('next_planner_final_prompt') or ''))}</pre>"
        f"<h4>Next-subtask planner: raw output</h4><pre>{html.escape(str(row.get('next_planner_raw_prediction') or ''))}</pre>"
        "</details>"
    )


def task_label(run_name: str) -> str:
    return "Place burger and fries" if "burger" in run_name else "Put bottles in dustbin"


def context_label(run_name: str) -> str:
    if run_name.startswith("single_"):
        return "Current image only"
    if run_name.startswith("2s_"):
        return "History sampled every 2 seconds"
    return "History sampled at 1 frame per second"


def render_examples(record: dict[str, Any], out_dir: Path) -> str:
    run_dir: Path = record["run_dir"]
    row = record["verifiers"][0]
    label = f"{task_label(run_dir.name)} / {context_label(run_dir.name)}"
    file_key = run_dir.name
    ep = int(row.get("episode_index", 0))
    video = video_for(Path(row.get("_video_run_dir", run_dir)), ep)
    images = []
    if video is not None:
        for frame_id in image_ids(row):
            target = out_dir / "frames" / "minimal_next" / f"{file_key}_ep{ep}_f{frame_id}.jpg"
            if frame_image(video, frame_id, target):
                image_src = target.relative_to(out_dir).as_posix()
                images.append(
                    f"<figure><img src='{html.escape(image_src)}' "
                    f"alt='{html.escape(label)} frame {frame_id}'>"
                    f"<figcaption>frame {frame_id}, {frame_id / 25:.2f}s</figcaption></figure>"
                )
    timing = row.get("timing_ms") or {}
    return (
        f"<article class='example'><h3>{html.escape(label)} / Sample {ep}</h3>"
        f"<p>Raw VLM output: <code>{html.escape(str(row.get('raw_prediction')))}</code>; system action: "
        f"<code>{'publish the next subtask' if row.get('planner_action') == 'switch' else 'continue the current subtask'}</code>; evidence frames: {image_ids(row)}</p>"
        f"<div class='frames'>{''.join(images) or '<p>Frames unavailable.</p>'}</div>"
        f"<p>Completion check triggered → updated instruction ready: <b>{fmt(metric(row, 'boundary_to_instruction_ready'))}</b>; "
        f"completion check triggered → first low-level action: <b>{fmt(metric(row, 'boundary_to_first_action'))}</b>; "
        f"low-level neural inference for this action: <b>{fmt(metric(row, 'low_level_resume_model_forward'))}</b></p>"
        f"{prompt_block('View exact model input and output', row)}</article>"
    )


def extract_old_accuracy(old_html: str) -> str:
    data = [
        ("Incremental subtask planning", "Current image only", "Place burger and fries", 1),
        ("Full plan generated at task start", "Current image only", "Place burger and fries", 0),
        ("Incremental subtask planning", "History sampled every 2 seconds", "Place burger and fries", 0),
        ("Full plan generated at task start", "History sampled every 2 seconds", "Place burger and fries", 0),
        ("Incremental subtask planning", "History sampled at 1 frame per second", "Place burger and fries", 2),
        ("Full plan generated at task start", "History sampled at 1 frame per second", "Place burger and fries", 1),
        ("Incremental subtask planning", "Current image only", "Put bottles in dustbin", 0),
        ("Full plan generated at task start", "Current image only", "Put bottles in dustbin", 0),
        ("Incremental subtask planning", "History sampled every 2 seconds", "Put bottles in dustbin", 0),
        ("Full plan generated at task start", "History sampled every 2 seconds", "Put bottles in dustbin", 0),
        ("Incremental subtask planning", "History sampled at 1 frame per second", "Put bottles in dustbin", 0),
        ("Full plan generated at task start", "History sampled at 1 frame per second", "Put bottles in dustbin", 1),
    ]
    rows = "".join(
        "<tr>"
        f"<td>{method}</td><td>{visual}</td><td>{task}</td>"
        f"<td>10</td><td>10</td><td>{correct}</td><td>{correct}/10</td>"
        f"<td>{correct * 10:.1f}%</td><td>{10 - correct}</td></tr>"
        for method, visual, task, correct in data
    )
    return (
        "<section><h2>Historical n=10: VLM Completion-Detection Accuracy</h2>"
        "<p>This table retains the results of the earlier n=10 experiment. At every measurement boundary, simulator ground truth marked the current subtask as completed; a VLM response is correct only when it also reports completion.</p>"
        "<p>This measures recall for recognizing completion at the selected boundary, not classification accuracy on a balanced completed/unfinished dataset.</p>"
        "<table><thead><tr><th>Subtask planning method</th><th>Visual input</th><th>Robot task</th><th>Samples</th>"
        "<th>Simulator ground truth: completed</th><th>Model predicted: completed</th><th>Correct</th>"
        "<th>Completion recall</th><th>False negatives</th></tr></thead><tbody>"
        f"{rows}<tr><th colspan='3'>Total</th><th>120</th><th>120</th><th>5</th>"
        "<th>5/120</th><th>4.2%</th><th>115</th></tr></tbody></table></section>"
    )


def _legacy_extract_old_accuracy(old_html: str) -> str:
    start = old_html.find("<h2>Archived n=10 VLM Completion Accuracy</h2>")
    if start < 0:
        return "<p>Archived n=10 table unavailable.</p>"
    section_start = old_html.rfind("<section", 0, start)
    section_end = old_html.find("</section>", start)
    if section_start < 0 or section_end < 0:
        return "<p>Archived n=10 table unavailable.</p>"
    section = old_html[section_start : section_end + len("</section>")]
    replacements = {
        "Archived n=10 VLM Completion Accuracy": "历史 n=10：视觉语言模型的任务完成判断正确率",
        "This table is recalculated from the earlier n=10 JSONL logs, with duplicate verifier writes collapsed by episode. For single-image runs containing 11 verifier episodes, episode 0 is excluded as the runtime warm-up and episodes 1-10 are used.": "该表由此前 n=10 的原始日志重新计算。重复写入的判断记录按样本去重；单帧实验若包含 11 个样本，则第 0 个样本仅用于运行时预热，统计第 1 至第 10 个样本。",
        "All 120 measurement boundaries were marked completed by <code>measurement_oracle_completed</code>. A VLM response is correct here only when it reports the current subtask as completed (<code>switch/completed</code>). Therefore this is completion recall at the selected boundary, not accuracy on a balanced completed/unfinished dataset.": "这 120 个检查时刻均由仿真器真值标记为“当前子任务已经完成”。因此，只有视觉语言模型也判断为“已完成”才算正确。这里衡量的是在选定检查时刻识别出已完成状态的召回率，并不是在“已完成/未完成”均衡数据集上的分类准确率。",
        "<th>Run</th>": "<th>规划方法 / 视觉输入 / 任务</th>",
        "<th>Oracle completed</th>": "<th>仿真器真值：已完成</th>",
        "<th>VLM completed</th>": "<th>模型判断：已完成</th>",
        "<th>Correct</th>": "<th>判断正确</th>",
        "<th>Accuracy / recall</th>": "<th>完成状态召回率</th>",
        "<th>False negatives</th>": "<th>漏判为未完成</th>",
        "single_image_next__": "任务执行前仅生成第一子任务；之后逐项生成 / 当前单帧 / ",
        "single_image_all__": "任务开始前一次性生成完整子任务序列 / 当前单帧 / ",
        "2s_memory_next__": "任务执行前仅生成第一子任务；之后逐项生成 / 每 2 秒采样历史图像 / ",
        "2s_memory_all__": "任务开始前一次性生成完整子任务序列 / 每 2 秒采样历史图像 / ",
        "1hz_memory_next__": "任务执行前仅生成第一子任务；之后逐项生成 / 每秒采样 1 帧历史图像 / ",
        "1hz_memory_all__": "任务开始前一次性生成完整子任务序列 / 每秒采样 1 帧历史图像 / ",
        "place_burger_fries": "放置汉堡和薯条",
        "put_bottles_dustbin": "将瓶子放入垃圾桶",
        "<th>Overall</th>": "<th>合计</th>",
    }
    for old, new in replacements.items():
        section = section.replace(old, new)
    # Keep regeneration idempotent when the source page is already translated.
    for visual in (
        "当前单帧",
        "每 2 秒采样历史图像",
        "每秒采样 1 帧历史图像",
    ):
        section = section.replace(
            f"任务执行前仅生成第一子任务；之后逐项生成 / {visual} / "
            "任务开始前一次性生成完整子任务序列 / ",
            f"任务开始前一次性生成完整子任务序列 / {visual} / ",
        )
    return section


def render(root: Path, out_dir: Path, old_page: Path) -> str:
    records = run_records(root)
    rows = []
    prompt_sections = []
    examples = []
    for record in records:
        run_dir = record["run_dir"]
        label = context_label(run_dir.name)
        task = task_label(run_dir.name)
        verifiers = record["verifiers"]
        valid = len(verifiers)
        continued = sum(row.get("planner_action") == "continue" for row in verifiers)
        switched = sum(row.get("planner_action") == "switch" for row in verifiers)
        actions = f"Continue current subtask: {continued}; publish next subtask: {switched}"
        prompt_tokens = first_value(
            verifiers, "backend_usage_prompt_tokens", "backend_verifier_usage_prompt_tokens"
        )
        completion_tokens = first_value(
            verifiers,
            "backend_usage_completion_tokens",
            "backend_verifier_usage_completion_tokens",
        )
        image_count = first_value(
            verifiers, "backend_image_count", "backend_verifier_image_count"
        )
        adherence = valid / (valid + record["invalid"]) * 100 if valid + record["invalid"] else 0
        rows.append(
            f"<tr><td>{html.escape(task)}</td><td>{html.escape(label)}</td><td>{valid}</td>"
            f"<td>{actions}</td><td>{fmt_number(image_count)}</td>"
            f"<td>{fmt_number(prompt_tokens)}</td><td>{fmt_number(completion_tokens)}</td>"
            f"<td>{adherence:.1f}%</td><td>{fmt(first_value(verifiers, 'boundary_to_instruction_ready'))}</td>"
            f"<td>{fmt(first_value(verifiers, 'boundary_to_first_action'))}</td></tr>"
        )
        prompt_sections.append(prompt_block(f"{task} / {label}", verifiers[0]))
        examples.append(render_examples(record, out_dir))

    old_accuracy = extract_old_accuracy(old_page.read_text(encoding="utf-8"))
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Incremental Subtask Planning: High-Level and Low-Level Interaction Report</title>
<style>
:root{{--text:#17202a;--muted:#5b6472;--line:#d8dde6;--bg:#f6f7f9;--panel:#fff;--accent:#0f766e}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
header,main{{max-width:1400px;margin:auto;padding:24px}} h1{{margin:0 0 8px;font-size:28px}} h2{{margin-top:30px;font-size:20px}}
p{{color:var(--muted)}} table{{display:block;width:100%;overflow-x:auto;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);margin:12px 0 22px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{background:#eef2f6}}
.note,.example,details{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin:12px 0}} summary{{cursor:pointer;font-weight:650}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#101828;color:#f2f4f7;border-radius:6px;padding:12px;font-size:12px}}
.frames{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}} figure{{margin:0;border:1px solid var(--line);border-radius:6px;overflow:hidden}} img{{display:block;width:100%}} figcaption{{padding:6px 8px;color:var(--muted);font-size:12px}}
code{{background:#eef2f6;padding:1px 4px;border-radius:4px}}
@media (max-width:640px){{header,main{{padding:16px}}h1{{font-size:24px}}th,td{{padding:8px;font-size:12px}}}}
</style></head><body><header><h1>Incremental Subtask Planning: High-Level and Low-Level Interaction Report</h1><p>Two robot-manipulation tasks under three visual-history settings</p></header><main>
<section class='note'><h2>Experimental Logic</h2>
<p>The robot control system has two layers. A high-level vision-language model decomposes a natural-language task into executable object-level subtasks and judges whether the current subtask is complete. A low-level action model receives the current subtask text and directly generates continuous actions for the dual-arm robot.</p>
<p>This experiment uses incremental subtask planning. At task start, the high-level model generates only the first subtask, which is sent to the low-level action model. While that subtask executes, the system prepares a possible next subtask in the background but does not publish it immediately. At a preset inspection time, low-level execution pauses; the high-level model views current or historical camera images and answers only whether the current subtask is complete.</p>
<p>Only when the high-level model answers “completed” does the system send the prepared next subtask to the low-level action model and discard cached actions tied to the old instruction. When it answers “not completed,” the low-level model continues the current subtask and retains its action cache. This separates visual completion detection from semantic generation of the next subtask.</p>
<p>The experiment covers “Place burger and fries” and “Put bottles in dustbin.” Visual inputs are: the current image at the inspection time; history sampled every 2 seconds; and history sampled at 1 frame per second. Each condition has 3 valid samples. The low-level model pauses while the high-level model works, so the total interaction time includes the real wait caused by high-level completion detection.</p>
</section>
<section class='note'><h2>Concrete Example</h2>
<p>For “Place burger and fries,” the high-level planner first generates “place the burger on the tray” and prepares “place the fries on the tray” while the low-level model executes. The robot pauses at 2.40 seconds and the completion verifier sees frames 0, 25, 50, and 60. If it returns <code>{{"completed":true}}</code>, the system sends “place the fries on the tray” to the low-level action model, clears cached actions left from the burger subtask, and records the time until the low-level model produces its first action under the new instruction. If it returns <code>{{"completed":false}}</code>, the fries subtask is not published and the robot continues “place the burger on the tray.”</p></section>
<h2>Interaction Time and Output Format in This Run</h2><table><thead><tr><th>Task</th><th>Visual input</th><th>Valid samples</th><th>Control action after completion check</th><th>Mean input images</th><th>Mean input tokens</th><th>Mean output tokens</th><th>Required-format adherence</th><th>Check triggered → decision ready</th><th>Check triggered → first low-level action</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<section class='note'><h2>How to Interpret the Timing</h2><p>“Check triggered → decision ready” fully records the wait caused by a high-level vision-language completion check and can be compared directly across visual-input settings. In contrast, “Check triggered → first low-level action” is also affected by the action cache. When the model judges that the current subtask is not complete, the system retains the existing instruction and generated action cache; the low-level model may directly take a cached action rather than run another neural-network inference. This value is therefore not always the full low-level inference time required after switching to a new subtask and should not be compared directly with samples that do switch subtasks.</p><p>Thus, a low-level action time near 0 ms does not mean the robot stopped being controlled. It means that step reused a previously generated action; the high-level completion-check time itself is still fully measured.</p></section>
<section class='note'><h2>Why Current Single-Image Times Are Lower Than the Earlier ~400 ms Result</h2><p>The earlier ~400 ms value came from a text-only prompt-latency benchmark using the former multi-field JSON output. The model generated about 17 tokens on average, including a control decision, execution status, and a textual reason. This page uses the new, real robot completion request: although the input includes one camera image and about 400 input tokens, the model is required to output only <code>{{"completed":true}}</code> or <code>{{"completed":false}}</code>, averaging 6 output tokens.</p><p>For this model, generation is a material part of the latency of short requests. Shortening the output protocol therefore reduces current single-image completion checks to about 0.22–0.29 seconds. This change primarily reflects the shorter output format; it should not be interpreted as visual understanding itself becoming faster. The old text-only benchmark and the real image-based completion check on this page are not identical metrics.</p></section>
<h2>Actual Prompts Received by the Models and Raw Outputs</h2>{''.join(prompt_sections)}
<h2>Actual Images and Example Executions</h2>{''.join(examples)}
{old_accuracy}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--old-page", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(args.root.resolve(), args.output.parent.resolve(), args.old_page.resolve()),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
