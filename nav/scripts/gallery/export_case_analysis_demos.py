"""Export publication-style navigation decision case-study figures.

The curated selections use real frames, odometry, observations, and reasoning
from recorded model runs.  Each page contains three cases in the same
left-to-right structure as the reference figure: minimap, egocentric RGB view,
and the model decision analysis.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 9
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS_SIZE = (2160, 1500)
ROW_HEIGHT = 478
MARGIN_X = 20
HEADER_HEIGHT = 66
PANEL_HEIGHT = 395
MINIMAP_WIDTH = 640
EGO_WIDTH = 400
GAP = 18
TEXT_WIDTH = (
    CANVAS_SIZE[0] - 2 * MARGIN_X - MINIMAP_WIDTH - EGO_WIDTH - 2 * GAP
)

FONT_ROOT = Path("/System/Library/Fonts/Supplemental")
FONT_REGULAR = FONT_ROOT / "Arial.ttf"
FONT_BOLD = FONT_ROOT / "Arial Bold.ttf"
FONT_ITALIC = FONT_ROOT / "Arial Bold Italic.ttf"
FONT_UI = Path("/System/Library/Fonts/Avenir Next.ttc")

COLORS = {
    "ink": "#111827",
    "muted": "#475569",
    "panel": "#f8fafc",
    "line": "#111111",
    "forward": "#059669",
    "turn left": "#2563eb",
    "turn right": "#dc2626",
    "stop": "#7c3aed",
    "sound": "#047857",
    "transition": "#b45309",
    "failure": "#b91c1c",
    "recovery": "#0369a1",
}


@dataclass(frozen=True)
class CaseSpec:
    scene: str
    point: str
    step: int
    title: str
    assessment: str
    assessment_kind: str
    analysis: str
    model_label: str = "GPT-5-mini"
    run_path: str | None = None


@dataclass(frozen=True)
class PageSpec:
    stem: str
    cases: tuple[CaseSpec, ...]


DEFAULT_PAGES = (
    PageSpec(
        stem="01_good_decisions",
        cases=(
            CaseSpec(
                "scene1",
                "point2",
                1,
                "Obstacle-aware reorientation",
                "SOUND DECISION",
                "sound",
                "The turn avoids a robot in the center lane and simultaneously reduces the large heading error to the goal.",
            ),
            CaseSpec(
                "scene15",
                "point3",
                11,
                "Reactive collision avoidance",
                "SOUND DECISION",
                "sound",
                "After a productive forward sequence, the newly blocked center view correctly triggers a turn instead of another forward step.",
            ),
            CaseSpec(
                "scene23",
                "point3",
                35,
                "Goal-aligned passage selection",
                "SOUND DECISION",
                "sound",
                "The agent remains goal-aligned while using the visible center-right passage; this run later finishes 0.77 m from the target.",
            ),
        ),
    ),
    PageSpec(
        stem="02_turn_oscillation",
        cases=(
            CaseSpec(
                "scene6",
                "point1",
                57,
                "Turn loop begins",
                "LOCALLY REASONABLE",
                "transition",
                "The forklift blocks the current view, so the first right turn is reasonable in isolation.",
            ),
            CaseSpec(
                "scene6",
                "point1",
                58,
                "Immediate reversal",
                "CONFLICT EMERGES",
                "transition",
                "Goal-bearing logic immediately reverses the avoidance turn, without first committing to forward progress through the clearer side.",
            ),
            CaseSpec(
                "scene6",
                "point1",
                59,
                "Repeated blocked view",
                "TURN OSCILLATION",
                "failure",
                "The agent returns to the same blocked heading and turns right again, creating a left-right loop with no translational progress.",
            ),
        ),
    ),
    PageSpec(
        stem="03_stuck_recovery",
        cases=(
            CaseSpec(
                "scene21",
                "point2",
                68,
                "Passability misjudgment",
                "RISKY FORWARD",
                "failure",
                "A low pallet dominates the near field, but the model treats the narrow center-right gap as passable and attempts forward motion.",
            ),
            CaseSpec(
                "scene21",
                "point2",
                69,
                "State-based blockage diagnosis",
                "CORRECT DIAGNOSIS",
                "recovery",
                "Unchanged world position reveals that the forward action failed, so the model correctly switches from translation to turning.",
            ),
            CaseSpec(
                "scene21",
                "point2",
                75,
                "Recovery fails to commit",
                "LOCAL LOOP",
                "failure",
                "Despite recognizing the lack of progress, alternating turns do not establish a new traversable heading and the episode remains stuck.",
            ),
        ),
    ),
)


CLOSED_MODEL_PAGES = (
    PageSpec(
        stem="01_gpt_5_6_sol",
        cases=(
            CaseSpec(
                "scene14", "point4", 41,
                "Pedestrian-triggered avoidance", "SOUND DECISION", "sound",
                "A close pedestrian interrupts an otherwise productive forward sequence; the turn preserves clearance while retaining a useful goalward component.",
                model_label="GPT-5.6 Sol",
                run_path="outputs/scene14/point4/gpt-5.6-sol/seed0",
            ),
            CaseSpec(
                "scene23", "point3", 38,
                "Collision-aware recovery", "CORRECT RECOVERY", "recovery",
                "The model uses the backward displacement in state history as evidence of contact and turns toward a clearer, target-compatible direction.",
                model_label="GPT-5.6 Sol",
                run_path="outputs/scene23/point3/gpt-5.6-sol/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 77,
                "Open-aisle hesitation", "LOCAL LOOP", "failure",
                "Although an aisle is visible to the right, another reorientation is selected at the same stationary pose; the episode ultimately ends 39.25 m away.",
                model_label="GPT-5.6 Sol",
                run_path="outputs/scene21/point2/gpt-5.6-sol/seed0",
            ),
        ),
    ),
    PageSpec(
        stem="02_gpt_6_astra",
        cases=(
            CaseSpec(
                "scene15", "point3", 5,
                "Cautious progress near a worker", "SOUND DECISION", "sound",
                "The model advances across clear near-field floor while explicitly maintaining clearance from the worker; this efficient run reaches 1.07 m in 24 steps.",
                model_label="GPT-6 Astra",
                run_path="analysis/case_analysis_demos/source_cache/gpt-6-astra/scene15/point3/seed0",
            ),
            CaseSpec(
                "scene14", "point3", 53,
                "Late-stage obstacle response", "SOUND DECISION", "sound",
                "Close pallet stacks replace the previously clear route, and Astra turns before contact; the episode later finishes only 0.60 m from the goal.",
                model_label="GPT-6 Astra",
                run_path="analysis/case_analysis_demos/source_cache/gpt-6-astra/scene14/point3/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 65,
                "Goal bias interrupts a detour", "LOOP PERSISTS", "failure",
                "The model recognizes recent oscillation but abandons the visibly open westbound aisle because it initially increases distance, prolonging the local loop.",
                model_label="GPT-6 Astra",
                run_path="analysis/case_analysis_demos/source_cache/gpt-6-astra/scene21/point2/seed0",
            ),
        ),
    ),
    PageSpec(
        stem="03_claude_sonnet_5",
        cases=(
            CaseSpec(
                "scene9", "point4", 3,
                "State-based blockage detection", "CORRECT DIAGNOSIS", "recovery",
                "The unchanged world pose overrides the apparently usable camera view, correctly identifying the conveyor ramp as a physical blockage.",
                model_label="Claude Sonnet 5",
                run_path="outputs/scene9/point4/claude-sonnet-5/seed0",
            ),
            CaseSpec(
                "scene24", "point2", 15,
                "Center-lane obstacle avoidance", "SOUND DECISION", "sound",
                "A person and forklift structure occupy the center lane, so the model turns instead of extending a long forward streak into the obstruction.",
                model_label="Claude Sonnet 5",
                run_path="outputs/scene24/point2/claude-sonnet-5/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 66,
                "Unsafe exploration action", "RISKY FORWARD", "failure",
                "Despite large angled surfaces occupying the immediate view and a long stationary history, the model tests forward and remains trapped.",
                model_label="Claude Sonnet 5",
                run_path="outputs/scene21/point2/claude-sonnet-5/seed0",
            ),
        ),
    ),
    PageSpec(
        stem="04_gemini_3_8_flash",
        cases=(
            CaseSpec(
                "scene2", "point4", 6,
                "Forklift-aware rerouting", "SOUND DECISION", "sound",
                "The model notices the forklift occupying the right side of the route and turns toward open floor rather than committing to a risky forward step.",
                model_label="Gemini 3.8 Flash",
                run_path="outputs/scene2/point4/gemini-3.8-flash/seed0",
            ),
            CaseSpec(
                "scene15", "point3", 10,
                "Mobile-obstacle response", "SOUND DECISION", "sound",
                "A pallet-carrying robot appears directly ahead after several successful moves; Gemini immediately turns toward the open side aisle.",
                model_label="Gemini 3.8 Flash",
                run_path="outputs/scene15/point3/gemini-3.8-flash/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 71,
                "Direction-sign inconsistency", "GOAL-LOGIC ERROR", "failure",
                "The view reports a clear aisle, but the model turns left from 0° while claiming this moves toward 180°; the episode ultimately ends 42.90 m away.",
                model_label="Gemini 3.8 Flash",
                run_path="outputs/scene21/point2/gemini-3.8-flash/seed0",
            ),
        ),
    ),
)


OPEN_MODEL_PAGES = (
    PageSpec(
        stem="01_glm_5_3_flash",
        cases=(
            CaseSpec(
                "scene15", "point3", 9,
                "Rack-aware aisle change", "SOUND DECISION", "sound",
                "After seven productive forward moves, the newly occluded center lane triggers a left turn into open floor; the run reaches 0.69 m in 24 steps.",
                model_label="GLM-5.3 Flash",
                run_path="outputs/scene15/point3/glm-5.3-flash/seed0",
            ),
            CaseSpec(
                "scene23", "point4", 26,
                "Dock-wall rejection near goal", "SOUND DECISION", "sound",
                "At 2.49 m from the goal, the closed dock door is correctly treated as impassable; the right turn enables the final approach and a 0.99 m finish.",
                model_label="GLM-5.3 Flash",
                run_path="outputs/scene23/point4/glm-5.3-flash/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 55,
                "Goal bias overrides an open retreat", "LOOP PERSISTS", "failure",
                "A traversable southbound retreat is directly visible, but target-bearing bias selects another in-place turn toward the known blocked face; the run ends 36.80 m away.",
                model_label="GLM-5.3 Flash",
                run_path="outputs/scene21/point2/glm-5.3-flash/seed0",
            ),
        ),
    ),
    PageSpec(
        stem="02_minimax_m3",
        cases=(
            CaseSpec(
                "scene20", "point1", 20,
                "Cabinet-aware final approach", "SOUND DECISION", "sound",
                "The cabinet-avoiding turn supports a successful 0.59 m finish.",
                model_label="MiniMax M3",
                run_path="outputs/scene20/point1/minimax-m3/seed0",
            ),
            CaseSpec(
                "scene13", "point1", 27,
                "State-grounded recovery", "CORRECT RECOVERY", "recovery",
                "After odometry exposes several ineffective moves in a tight gap, the model commits to the visibly open north aisle and ultimately recovers to a 0.73 m finish.",
                model_label="MiniMax M3",
                run_path="outputs/scene13/point1/minimax-m3/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 76,
                "Visual clearance overrules contact history", "RISKY FORWARD", "failure",
                "The corridor looks open at eye level, but repeated unchanged poses show that the low grating blocks motion; another forward attempt repeats the failure and the run ends 35.60 m away.",
                model_label="MiniMax M3",
                run_path="outputs/scene21/point2/minimax-m3/seed0",
            ),
        ),
    ),
    PageSpec(
        stem="03_deepseek_v4_flash",
        cases=(
            CaseSpec(
                "scene20", "point1", 14,
                "Rack-triggered rerouting", "SOUND DECISION", "sound",
                "A rack fills the forward path, and the selected right turn both avoids contact and opens the successful detour; the episode later reaches 0.92 m.",
                model_label="DeepSeek V4 Flash",
                run_path="outputs/_api_kiro_v1/scene20/point1/deepseek-v4-flash-vision-exp/seed0",
            ),
            CaseSpec(
                "scene19", "point2", 15,
                "Detour commitment after blockage", "CORRECT RECOVERY", "recovery",
                "After routing around the red container, the model commits to the newly visible aisle rather than undoing its turn; this recovered trajectory finishes 1.16 m from the goal.",
                model_label="DeepSeek V4 Flash",
                run_path="outputs/_api_kiro_v1/scene19/point2/deepseek-v4-flash-vision-exp/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 74,
                "Repeated local avoidance", "LOOP PERSISTS", "failure",
                "The right turn is locally sensible, but the same pallet reappears at an unchanged pose; without committing to the visible side aisle, the episode remains 35.88 m away.",
                model_label="DeepSeek V4 Flash",
                run_path="outputs/_api_kiro_v1/scene21/point2/deepseek-v4-flash-vision-exp/seed0",
            ),
        ),
    ),
    PageSpec(
        stem="04_qwen_3_8_flash",
        cases=(
            CaseSpec(
                "scene2", "point3", 49,
                "Pallet-aware route correction", "SOUND DECISION", "sound",
                "The pallet row closes the forward lane, so Qwen turns before contact and subsequently follows the cleared route to a 1.33 m finish.",
                model_label="Qwen 3.8 Flash",
                run_path="outputs/_api_kiro_v1/scene2/point3/qwen3.8-flash/seed0",
            ),
            CaseSpec(
                "scene14", "point4", 69,
                "Pedestrian-aware pause", "DYNAMIC SAFETY", "sound",
                "A crossing worker triggers a temporary stop at 3.02 m; once the person clears the center lane, the next forward move completes the approach at 1.52 m.",
                model_label="Qwen 3.8 Flash",
                run_path="outputs/_api_kiro_v1/scene14/point4/qwen3.8-flash/seed0",
            ),
            CaseSpec(
                "scene21", "point2", 69,
                "Recognized loop, repeated action", "LOCAL OSCILLATION", "failure",
                "The reasoning explicitly identifies the left-right loop, yet selects the same reversal again; the pose remains unchanged through the episode and ends 41.66 m away.",
                model_label="Qwen 3.8 Flash",
                run_path="outputs/_api_kiro_v1/scene21/point2/qwen3.8-flash/seed0",
            ),
        ),
    ),
)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


FONTS = {
    "header_bold": font(FONT_ITALIC, 28),
    "header": font(FONT_REGULAR, 27),
    "meta": font(FONT_BOLD, 28),
    "body": font(FONT_REGULAR, 23),
    "body_bold": font(FONT_BOLD, 23),
    "badge": font(FONT_BOLD, 19),
    "small": font(FONT_REGULAR, 18),
}

FIGURE_SIZE_IN = (14.4, 10.0)
VECTOR_FONTS = {
    "header_bold": FontProperties(fname=str(FONT_ITALIC), size=13.8),
    "header": FontProperties(fname=str(FONT_REGULAR), size=14.5),
    "odometry_label": FontProperties(fname=str(FONT_UI), size=8.8, weight="bold"),
    "odometry_value": FontProperties(fname=str(FONT_UI), size=12.2, weight="bold"),
    "meta": FontProperties(fname=str(FONT_BOLD), size=16.5),
    "body": FontProperties(fname=str(FONT_REGULAR), size=14.2),
    "body_bold": FontProperties(fname=str(FONT_BOLD), size=14.2),
    "badge": FontProperties(fname=str(FONT_BOLD), size=9.5),
    "panel_model": FontProperties(fname=str(FONT_BOLD), size=14.8),
    "panel_pill": FontProperties(fname=str(FONT_BOLD), size=10.2),
    "small": FontProperties(fname=str(FONT_REGULAR), size=8.8),
}


def read_action_row(run_dir: Path, step: int) -> dict[str, str]:
    with (run_dir / "llm_actions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) == step:
                return row
    raise ValueError(f"Step {step} is absent from {run_dir / 'llm_actions.csv'}")


QA_BLOCK = re.compile(
    r"^=== Step (?P<step>\d+) ===.*?^\[A\]\n"
    r"Observation: (?P<observation>.*?)\n"
    r"Action: (?P<action>.*?)\n"
    r"Reasoning: (?P<reasoning>.*?)\n=+",
    re.MULTILINE | re.DOTALL,
)


def read_qa(run_dir: Path, step: int) -> dict[str, str]:
    content = (run_dir / "agent_qa.txt").read_text(encoding="utf-8")
    for match in QA_BLOCK.finditer(content):
        if int(match.group("step")) == step:
            return {
                key: " ".join(match.group(key).split())
                for key in ("observation", "action", "reasoning")
            }
    raise ValueError(f"Step {step} is absent from {run_dir / 'agent_qa.txt'}")


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=text_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped_labeled_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    text: str,
    max_width: int,
    line_height: int = 29,
) -> int:
    x, y = xy
    label_width = int(draw.textlength(label, font=FONTS["body_bold"]))
    first_width = max_width - label_width
    words = text.split()
    first = ""
    consumed = 0
    for index, word in enumerate(words):
        candidate = f"{first} {word}".strip()
        if not first or draw.textlength(candidate, font=FONTS["body"]) <= first_width:
            first = candidate
            consumed = index + 1
        else:
            break
    draw.text((x, y), label, fill=COLORS["ink"], font=FONTS["body_bold"])
    draw.text(
        (x + label_width, y),
        first,
        fill=COLORS["ink"],
        font=FONTS["body"],
    )
    y += line_height
    remainder = " ".join(words[consumed:])
    for line in wrap_text(draw, remainder, FONTS["body"], max_width):
        draw.text((x, y), line, fill=COLORS["ink"], font=FONTS["body"])
        y += line_height
    return y


def draw_header(
    draw: ImageDraw.ImageDraw,
    spec: CaseSpec,
    row: dict[str, str],
    y: int,
) -> None:
    prefix = f"{spec.title} · "
    position = f"({float(row['curr_world_x']):.1f}, {float(row['curr_world_z']):.1f}) m"
    target = f"({float(row['target_world_x']):.1f}, {float(row['target_world_z']):.1f}) m"
    heading = float(row["curr_direction_y"]) % 360
    distance = float(row["distance_world"])
    suffix = (
        f"Global Odometry: Position {position}, θ={heading:.1f}°, "
        f"Target {target}, Distance {distance:.2f} m"
    )
    prefix_width = draw.textlength(prefix, font=FONTS["header_bold"])
    suffix_width = draw.textlength(suffix, font=FONTS["header"])
    x = (CANVAS_SIZE[0] - prefix_width - suffix_width) / 2
    draw.text((x, y), prefix, fill=COLORS["ink"], font=FONTS["header_bold"])
    draw.text(
        (x + prefix_width, y),
        suffix,
        fill=COLORS["ink"],
        font=FONTS["header"],
    )


def draw_text_panel(
    draw: ImageDraw.ImageDraw,
    spec: CaseSpec,
    qa: dict[str, str],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=22, fill=COLORS["panel"], outline=COLORS["line"], width=4)
    x = x0 + 24
    y = y0 + 18
    model_text = spec.model_label
    draw.text((x, y), model_text, fill=COLORS["ink"], font=FONTS["meta"])
    cursor = x + draw.textlength(model_text, font=FONTS["meta"]) + 28
    draw.text((cursor, y), "Action:", fill=COLORS["ink"], font=FONTS["meta"])
    cursor += draw.textlength("Action:", font=FONTS["meta"]) + 12
    action_color = COLORS.get(qa["action"].lower(), COLORS["ink"])
    draw.text((cursor, y), qa["action"], fill=action_color, font=FONTS["meta"])

    badge_color = COLORS[spec.assessment_kind]
    badge_width = int(draw.textlength(spec.assessment, font=FONTS["badge"])) + 30
    badge_box = (x1 - badge_width - 18, y0 + 17, x1 - 18, y0 + 50)
    draw.rounded_rectangle(badge_box, radius=15, fill=badge_color)
    draw.text(
        (badge_box[0] + 15, badge_box[1] + 5),
        spec.assessment,
        fill="white",
        font=FONTS["badge"],
    )

    y += 51
    usable_width = x1 - x - 24
    y = draw_wrapped_labeled_text(
        draw, (x, y), "Observation: ", qa["observation"], usable_width
    )
    y += 7
    y = draw_wrapped_labeled_text(
        draw, (x, y), "Reasoning: ", qa["reasoning"], usable_width
    )
    y += 7
    draw_wrapped_labeled_text(
        draw, (x, y), "Analysis: ", spec.analysis, usable_width
    )

    footer = f"{spec.scene} / {spec.point} / step {spec.step} · real recorded model output"
    footer_width = draw.textlength(footer, font=FONTS["small"])
    draw.text(
        (x1 - footer_width - 20, y1 - 28),
        footer,
        fill=COLORS["muted"],
        font=FONTS["small"],
    )


def render_page(root: Path, page: PageSpec) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, spec in enumerate(page.cases):
        run_dir = (
            root / spec.run_path
            if spec.run_path is not None
            else root / spec.scene / spec.point / "gpt-5-mini" / "seed0"
        )
        action_row = read_action_row(run_dir, spec.step)
        qa = read_qa(run_dir, spec.step)
        frame = spec.step - 1
        minimap_path = run_dir / "llm_minimap_target" / f"{frame}.png"
        ego_path = run_dir / "llm_fp" / f"{frame}.png"
        if not minimap_path.exists() or not ego_path.exists():
            raise FileNotFoundError(
                f"Missing frame for {spec.scene}/{spec.point}/step {spec.step}: "
                f"{minimap_path}, {ego_path}"
            )

        row_y = row_index * ROW_HEIGHT
        draw_header(draw, spec, action_row, row_y + 13)
        panel_y = row_y + HEADER_HEIGHT
        minimap_box = (MARGIN_X, panel_y, MARGIN_X + MINIMAP_WIDTH, panel_y + PANEL_HEIGHT)
        ego_x = minimap_box[2] + GAP
        ego_box = (ego_x, panel_y, ego_x + EGO_WIDTH, panel_y + PANEL_HEIGHT)
        text_x = ego_box[2] + GAP
        text_box = (text_x, panel_y, text_x + TEXT_WIDTH, panel_y + PANEL_HEIGHT)

        minimap = fit_image(minimap_path, (MINIMAP_WIDTH, PANEL_HEIGHT))
        ego = fit_image(ego_path, (EGO_WIDTH, PANEL_HEIGHT))
        canvas.paste(minimap, minimap_box[:2])
        canvas.paste(ego, ego_box[:2])
        draw.rectangle(minimap_box, outline=COLORS["line"], width=4)
        draw.rectangle(ego_box, outline=COLORS["line"], width=4)
        draw_text_panel(draw, spec, qa, text_box)

    return canvas


def _figure_box(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """Convert top-left pixel-layout coordinates to figure fractions."""
    x0, y0, x1, y1 = box
    return (
        x0 / CANVAS_SIZE[0],
        1.0 - y1 / CANVAS_SIZE[1],
        (x1 - x0) / CANVAS_SIZE[0],
        (y1 - y0) / CANVAS_SIZE[1],
    )


def _crop_to_aspect(path: Path, target_aspect: float) -> Image.Image:
    """Crop without upsampling so PDFs retain the original sensor pixels."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    current_aspect = width / height
    if current_aspect > target_aspect:
        new_width = max(1, round(height * target_aspect))
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    new_height = max(1, round(width / target_aspect))
    top = (height - new_height) // 2
    return image.crop((0, top, width, top + new_height))


def _text_width_fraction(text: str, text_font: FontProperties) -> float:
    if not text:
        return 0.0
    width_points = TextPath((0, 0), text, prop=text_font).get_extents().width
    return width_points / (72.0 * FIGURE_SIZE_IN[0])


def _add_vector_image(
    figure: plt.Figure,
    path: Path,
    box: tuple[int, int, int, int],
) -> None:
    x, y, width, height = _figure_box(box)
    axes = figure.add_axes((x, y, width, height), zorder=1)
    image = _crop_to_aspect(path, (box[2] - box[0]) / (box[3] - box[1]))
    axes.imshow(image, interpolation="lanczos", aspect="auto")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(COLORS["line"])
        spine.set_linewidth(1.5)


def _draw_vector_header(
    figure: plt.Figure,
    spec: CaseSpec,
    row: dict[str, str],
    row_y: int,
) -> None:
    position = f"({float(row['curr_world_x']):.1f}, {float(row['curr_world_z']):.1f}) m"
    target = f"({float(row['target_world_x']):.1f}, {float(row['target_world_z']):.1f}) m"
    heading = float(row["curr_direction_y"]) % 360
    position_gap = 18
    compact_gap = 8
    group_gap = 4
    position_width = 170
    compact_width = 95
    metrics = (
        ("CURRENT (X, Z)", position, position_width),
        ("TARGET (X, Z)", target, position_width),
        ("HEADING", f"{heading:.1f}°", compact_width),
        ("DISTANCE", f"{float(row['distance_world']):.2f} m", compact_width),
    )
    metrics_width_pixels = (
        2 * position_width
        + 2 * compact_width
        + position_gap
        + compact_gap
        + group_gap
    )
    available_width = metrics_width_pixels / CANVAS_SIZE[0]
    column_left = MARGIN_X / CANVAS_SIZE[0]
    label_y = 1.0 - (row_y + 18) / CANVAS_SIZE[1]
    value_y = 1.0 - (row_y + 41) / CANVAS_SIZE[1]
    for metric_index, (label, value, width_pixels) in enumerate(metrics):
        column_width = width_pixels / CANVAS_SIZE[0]
        center_x = column_left + column_width / 2
        figure.text(
            center_x,
            label_y,
            label,
            color="#64748b",
            fontproperties=VECTOR_FONTS["odometry_label"],
            ha="center",
            va="center",
            zorder=2,
        )
        figure.text(
            center_x,
            value_y,
            value,
            color=COLORS["ink"],
            fontproperties=VECTOR_FONTS["odometry_value"],
            ha="center",
            va="center",
            zorder=2,
        )
        column_left += column_width
        if metric_index == 0:
            column_left += position_gap / CANVAS_SIZE[0]
        elif metric_index == 1:
            column_left += group_gap / CANVAS_SIZE[0]
        elif metric_index == 2:
            column_left += compact_gap / CANVAS_SIZE[0]

    rule_x = MARGIN_X / CANVAS_SIZE[0]
    rule_y = 1.0 - (row_y + 57) / CANVAS_SIZE[1]
    rule = FancyBboxPatch(
        (rule_x, rule_y - 0.00035),
        available_width,
        0.0007,
        boxstyle="square,pad=0",
        transform=figure.transFigure,
        facecolor="#dbe2ea",
        edgecolor="none",
        zorder=1,
    )
    figure.add_artist(rule)

    ego_left = MARGIN_X + MINIMAP_WIDTH + GAP
    ego_center_x = (ego_left + EGO_WIDTH / 2) / CANVAS_SIZE[0]
    y = 1.0 - (row_y + 33) / CANVAS_SIZE[1]
    figure.text(
        ego_center_x,
        y,
        spec.title,
        color=COLORS["ink"],
        fontproperties=VECTOR_FONTS["header_bold"],
        ha="center",
        va="center",
    )


def _body_layout(
    paragraphs: tuple[tuple[str, str], ...],
    available_height: float,
    max_width: float,
) -> tuple[FontProperties, FontProperties, list[tuple[str, list[str]]], float]:
    """Choose the largest body size that fits the right-hand vector panel."""
    figure_height = FIGURE_SIZE_IN[1]
    for size in (14.2, 13.8, 13.4, 13.0, 12.6, 12.2, 11.8):
        body_font = FontProperties(fname=str(FONT_REGULAR), size=size)
        bold_font = FontProperties(fname=str(FONT_BOLD), size=size)
        wrapped: list[tuple[str, list[str]]] = []
        for label, text in paragraphs:
            label_width = _text_width_fraction(label, bold_font) + 0.004
            words = text.split()
            lines: list[str] = []
            current = ""
            line_limit = max_width - label_width
            for word in words:
                candidate = f"{current} {word}".strip()
                if not current or _text_width_fraction(candidate, body_font) <= line_limit:
                    current = candidate
                    continue
                lines.append(current)
                current = word
                line_limit = max_width
            if current:
                lines.append(current)
            wrapped.append((label, lines))
        line_step = size * 1.14 / (72.0 * figure_height)
        required = sum(len(lines) * line_step for _, lines in wrapped)
        if required <= available_height or size == 11.8:
            return (
                body_font,
                bold_font,
                wrapped,
                line_step,
            )
    raise AssertionError("unreachable")


def _draw_vector_text_panel(
    figure: plt.Figure,
    spec: CaseSpec,
    qa: dict[str, str],
    box: tuple[int, int, int, int],
) -> None:
    x, y, width, height = _figure_box(box)
    shadow = FancyBboxPatch(
        (x + 0.0025, y - 0.0025),
        width,
        height,
        boxstyle="round,pad=0.002,rounding_size=0.010",
        transform=figure.transFigure,
        facecolor="#e2e8f0",
        edgecolor="none",
        zorder=0,
    )
    figure.add_artist(shadow)
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.002,rounding_size=0.010",
        transform=figure.transFigure,
        facecolor="white",
        edgecolor="#cbd5e1",
        linewidth=1.0,
        zorder=1,
    )
    figure.add_artist(panel)

    inset = 0.006
    header_height = 0.043
    header_x = x + inset
    header_y = y + height - inset - header_height
    header_width = width - 2 * inset
    header = FancyBboxPatch(
        (header_x, header_y),
        header_width,
        header_height,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        transform=figure.transFigure,
        facecolor="#475569",
        edgecolor="none",
        zorder=2,
    )
    figure.add_artist(header)

    header_center = header_y + header_height / 2
    left = header_x + 0.010
    figure.text(
        left,
        header_center,
        spec.model_label,
        color="white",
        fontproperties=VECTOR_FONTS["panel_model"],
        va="center",
        zorder=3,
    )
    cursor = left + _text_width_fraction(
        spec.model_label, VECTOR_FONTS["panel_model"]
    ) + 0.014
    action_color = COLORS.get(qa["action"].lower(), "#334155")
    figure.text(
        cursor,
        header_center,
        f"ACTION  ·  {qa['action'].upper()}",
        color="white",
        fontproperties=VECTOR_FONTS["panel_pill"],
        va="center",
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.9",
            "facecolor": action_color,
            "edgecolor": "none",
        },
        zorder=3,
    )
    figure.text(
        header_x + header_width - 0.010,
        header_center,
        spec.assessment,
        color="white",
        fontproperties=VECTOR_FONTS["badge"],
        ha="right",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.9",
            "facecolor": COLORS[spec.assessment_kind],
            "edgecolor": "none",
        },
        zorder=3,
    )

    body_top = header_y - 0.006
    body_bottom = y + 0.024
    card_gap = 0.004
    card_pad_y = 0.003
    paragraphs = (
        ("Observation:", qa["observation"]),
        ("Reasoning:", qa["reasoning"]),
        ("Analysis:", spec.analysis),
    )
    available_text_height = (
        body_top
        - body_bottom
        - 2 * card_pad_y * len(paragraphs)
        - card_gap * (len(paragraphs) - 1)
    )
    card_x = x + 0.008
    card_width = width - 0.016
    text_x = card_x + 0.013
    text_width = card_x + card_width - text_x - 0.008
    body_font, bold_font, wrapped, line_step = _body_layout(
        paragraphs, available_text_height, text_width
    )
    analysis_background = {
        "failure": "#fef2f2",
        "transition": "#fffbeb",
    }.get(spec.assessment_kind, "#f0fdf4")
    card_styles = (
        ("#eff6ff", "#2563eb"),
        ("#f5f3ff", "#7c3aed"),
        (analysis_background, COLORS[spec.assessment_kind]),
    )
    card_top = body_top
    for (label, lines), (card_color, accent_color) in zip(wrapped, card_styles):
        card_height = len(lines) * line_step + 2 * card_pad_y
        card_y = card_top - card_height
        card = FancyBboxPatch(
            (card_x, card_y),
            card_width,
            card_height,
            boxstyle="round,pad=0.001,rounding_size=0.006",
            transform=figure.transFigure,
            facecolor=card_color,
            edgecolor="none",
            zorder=2,
        )
        figure.add_artist(card)
        accent = FancyBboxPatch(
            (card_x + 0.003, card_y + 0.003),
            0.0035,
            card_height - 0.006,
            boxstyle="round,pad=0,rounding_size=0.002",
            transform=figure.transFigure,
            facecolor=accent_color,
            edgecolor="none",
            zorder=3,
        )
        figure.add_artist(accent)
        paragraph_y = card_top - card_pad_y
        label_width = _text_width_fraction(label, bold_font) + 0.004
        figure.text(
            text_x,
            paragraph_y,
            label,
            color=accent_color,
            fontproperties=bold_font,
            va="top",
            zorder=4,
        )
        for line_index, line in enumerate(lines):
            figure.text(
                text_x + (label_width if line_index == 0 else 0.0),
                paragraph_y - line_index * line_step,
                line,
                color=COLORS["ink"],
                fontproperties=body_font,
                va="top",
                zorder=3,
            )
        card_top = card_y - card_gap

    footer = f"{spec.scene} / {spec.point} / step {spec.step} · real recorded model output"
    figure.text(
        x + width - 0.008,
        y + 0.009,
        footer,
        color=COLORS["muted"],
        fontproperties=VECTOR_FONTS["small"],
        ha="right",
        va="bottom",
        zorder=2,
    )


def render_vector_page(root: Path, page: PageSpec) -> plt.Figure:
    """Build a mixed vector/raster page without flattening text into an image."""
    figure = plt.figure(figsize=FIGURE_SIZE_IN, facecolor="white")
    for row_index, spec in enumerate(page.cases):
        run_dir = (
            root / spec.run_path
            if spec.run_path is not None
            else root / spec.scene / spec.point / "gpt-5-mini" / "seed0"
        )
        action_row = read_action_row(run_dir, spec.step)
        qa = read_qa(run_dir, spec.step)
        frame = spec.step - 1
        minimap_path = run_dir / "llm_minimap_target" / f"{frame}.png"
        ego_path = run_dir / "llm_fp" / f"{frame}.png"
        if not minimap_path.exists() or not ego_path.exists():
            raise FileNotFoundError(
                f"Missing frame for {spec.scene}/{spec.point}/step {spec.step}: "
                f"{minimap_path}, {ego_path}"
            )

        row_y = row_index * ROW_HEIGHT
        _draw_vector_header(figure, spec, action_row, row_y)
        panel_y = row_y + HEADER_HEIGHT
        minimap_box = (
            MARGIN_X,
            panel_y,
            MARGIN_X + MINIMAP_WIDTH,
            panel_y + PANEL_HEIGHT,
        )
        ego_x = minimap_box[2] + GAP
        ego_box = (ego_x, panel_y, ego_x + EGO_WIDTH, panel_y + PANEL_HEIGHT)
        text_x = ego_box[2] + GAP
        text_box = (text_x, panel_y, text_x + TEXT_WIDTH, panel_y + PANEL_HEIGHT)
        _add_vector_image(figure, minimap_path, minimap_box)
        _add_vector_image(figure, ego_path, ego_box)
        _draw_vector_text_panel(figure, spec, qa, text_box)
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        choices=("gpt5mini", "closed-models", "open-models", "astra-single", "all"),
        default="gpt5mini",
        help="Which curated case-study collection to export.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/_api_kiro_v1"),
        help="Root containing scene*/point*/gpt-5-mini/seed0 runs.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used by the closed-model case paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory when exporting one collection.",
    )
    return parser.parse_args()


def export_pages(
    root: Path,
    pages: tuple[PageSpec, ...],
    output_dir: Path,
    combined_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / combined_name
    metadata = {
        "Title": "IndustryNav Navigation Decision Case Studies",
        "Author": "IndustryNav",
        "Subject": "Real navigation observations, actions, and reasoning",
    }
    with PdfPages(combined_path, metadata=metadata) as combined_pdf:
        for page in pages:
            figure = render_vector_page(root, page)
            png_path = output_dir / f"{page.stem}.png"
            pdf_path = output_dir / f"{page.stem}.pdf"
            figure.savefig(
                png_path,
                format="png",
                dpi=150,
                facecolor="white",
            )
            figure.savefig(
                pdf_path,
                format="pdf",
                facecolor="white",
                metadata=metadata,
            )
            combined_pdf.savefig(figure, facecolor="white")
            plt.close(figure)
            print(f"Wrote {png_path}")
            print(f"Wrote {pdf_path}")
    print(f"Wrote {combined_path}")


def export_single_case_pdf(root: Path, spec: CaseSpec, output_path: Path) -> None:
    """Export one case as a compact one-row vector PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page = PageSpec(stem=output_path.stem, cases=(spec,))
    figure = render_vector_page(root, page)
    row_height_inches = FIGURE_SIZE_IN[1] * ROW_HEIGHT / CANVAS_SIZE[1]
    crop = Bbox.from_bounds(
        0,
        FIGURE_SIZE_IN[1] - row_height_inches,
        FIGURE_SIZE_IN[0],
        row_height_inches,
    )
    metadata = {
        "Title": f"IndustryNav {spec.model_label} Navigation Case Study",
        "Author": "IndustryNav",
        "Subject": "Real navigation observation, action, and reasoning",
    }
    figure.savefig(
        output_path,
        format="pdf",
        facecolor="white",
        bbox_inches=crop,
        pad_inches=0,
        metadata=metadata,
    )
    plt.close(figure)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    if args.output_dir is not None and args.collection == "all":
        raise SystemExit("--output-dir cannot be combined with --collection all")
    if args.collection in {"gpt5mini", "all"}:
        export_pages(
            args.input_root,
            DEFAULT_PAGES,
            args.output_dir or Path("analysis/case_analysis_demos/gpt5mini"),
            "case_gpt5mini_demos.pdf",
        )
    if args.collection in {"closed-models", "all"}:
        export_pages(
            args.repo_root,
            CLOSED_MODEL_PAGES,
            args.output_dir or Path("analysis/case_analysis_demos/closed_models"),
            "recent_closed_model_case_demos.pdf",
        )
    if args.collection in {"open-models", "all"}:
        export_pages(
            args.repo_root,
            OPEN_MODEL_PAGES,
            args.output_dir or Path("analysis/case_analysis_demos/open_models"),
            "open_model_case_demos.pdf",
        )
    if args.collection == "astra-single":
        output_path = (
            args.output_dir / "gpt_6_astra_single_case.pdf"
            if args.output_dir is not None
            else Path(
                "analysis/case_analysis_demos/closed_models/"
                "gpt_6_astra_single_case.pdf"
            )
        )
        export_single_case_pdf(
            args.repo_root,
            CLOSED_MODEL_PAGES[1].cases[1],
            output_path,
        )


if __name__ == "__main__":
    main()
