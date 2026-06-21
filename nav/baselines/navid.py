import os
import random
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

from nav.config import NAVID_DEFAULTS


class NaVidBaseline:
    """Thin adapter for the public NaVid VLN-CE checkpoint interface.

    The heavy model code stays in an external NaVid checkout. Point this class
    at that checkout with ``repo_path`` and at the checkpoint with
    ``model_path``.
    """

    def __init__(
        self,
        model_path: str,
        repo_path: str = "",
        instruction: str = "",
        max_new_tokens: int = NAVID_DEFAULTS.max_new_tokens,
        temperature: float = NAVID_DEFAULTS.temperature,
    ):
        if not model_path:
            raise ValueError("NaVid requires --navid_model_path or NAVID_MODEL_PATH.")

        if repo_path:
            repo_path = os.path.abspath(repo_path)
            if not os.path.isdir(repo_path):
                raise FileNotFoundError(f"NaVid repo path does not exist: {repo_path}")
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            import torch
            from navid.constants import (  # type: ignore
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from navid.conversation import SeparatorStyle, conv_templates  # type: ignore
            from navid.mm_utils import (  # type: ignore
                KeywordsStoppingCriteria,
                get_model_name_from_path,
                tokenizer_image_token,
            )
            from navid.model.builder import load_pretrained_model  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import NaVid. Set NAVID_REPO or --navid_repo to a "
                "checkout of https://github.com/jzhzhang/NaVid-VLN-CE and make "
                "sure its Python dependencies are installed in this environment."
            ) from exc

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.tokenizer_image_token = tokenizer_image_token

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, None, self.model_name)
        )
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.conv_mode = NAVID_DEFAULTS.conv_mode
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.instruction = instruction.strip() or NAVID_DEFAULTS.instruction
        self.history_rgb_tensor = None
        self.rgb_list: List[np.ndarray] = []
        self.pending_actions: List[str] = []

    def reset(self) -> None:
        self.history_rgb_tensor = None
        self.rgb_list = []
        self.pending_actions = []

    def decide(self, ego_rgb: Optional[np.ndarray]) -> Tuple[str, str]:
        if self.pending_actions:
            action = self.pending_actions.pop(0)
            return action, f"NaVid pending action: {action}"

        if ego_rgb is None:
            action = random.choice(["forward", "turn left", "turn right"])
            return action, "NaVid missing ego RGB; random fallback."

        self.rgb_list.append(self._to_uint8_rgb(ego_rgb))
        prompt = self._build_prompt()
        output = self._predict(prompt)
        self.pending_actions = self._parse_actions(output)
        if not self.pending_actions:
            self.pending_actions = [random.choice(["forward", "turn left", "turn right"])]
        action = self.pending_actions.pop(0)
        return action, output

    def _build_prompt(self) -> str:
        return (
            "You are controlling a mobile robot for visual navigation. "
            "Use the sequence of past camera views and the current camera view. "
            f"Task: {self.instruction} "
            "Reply with one navigation command such as stop, move forward a distance, "
            "turn left by degrees, or turn right by degrees."
        )

    def _process_images(self):
        start_idx = 0
        if self.history_rgb_tensor is not None:
            start_idx = int(self.history_rgb_tensor.shape[0])

        batch = np.asarray(self.rgb_list[start_idx:])
        video = self.image_processor.preprocess(batch, return_tensors="pt")[
            "pixel_values"
        ]
        video = video.half().cuda()
        if self.history_rgb_tensor is None:
            self.history_rgb_tensor = video
        else:
            self.history_rgb_tensor = self.torch.cat(
                (self.history_rgb_tensor, video), dim=0
            )
        return [self.history_rgb_tensor]

    def _predict(self, question: str) -> str:
        torch = self.torch
        qs = question
        if self.model.config.mm_use_im_start_end:
            qs = (
                self.DEFAULT_IM_START_TOKEN
                + self.DEFAULT_IMAGE_TOKEN
                + self.DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = self.tokenizer_image_token(
            prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).cuda()
        input_ids = self._inject_navigation_tokens(token_prompt).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = self.KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )
        images = self._process_images()

        with torch.inference_mode():
            if hasattr(self.model, "update_prompt"):
                self.model.update_prompt([[question]])
            output_ids = self.model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_len = input_ids.shape[1]
        output = self.tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)].strip()
        return output

    def _inject_navigation_tokens(self, token_prompt):
        torch = self.torch
        special_tokens = [
            "",
            "",
            "",
            "",
            "[Navigation]",
            "",
        ]
        video_start, video_end, image_start, image_end, navigation, separator = [
            self.tokenizer(token, return_tensors="pt").input_ids[0][1:].cuda()
            for token in special_tokens
        ]

        chunks = []
        indices = torch.where(token_prompt == self.IMAGE_TOKEN_INDEX)[0]
        while indices.numel() > 0:
            idx = indices[0]
            chunks.append(token_prompt[:idx])
            chunks.extend(
                [
                    video_start,
                    separator,
                    token_prompt[idx : idx + 1],
                    video_end,
                    image_start,
                    image_end,
                    navigation,
                ]
            )
            token_prompt = token_prompt[idx + 1 :]
            indices = torch.where(token_prompt == self.IMAGE_TOKEN_INDEX)[0]

        if token_prompt.numel() > 0:
            chunks.append(token_prompt)
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _parse_actions(output: str) -> List[str]:
        text = output.strip().lower()
        if "stop" in text:
            return ["stop"]

        if "forward" in text:
            steps = NaVidBaseline._extract_repeats(text, divisor=25, default=1)
            return ["forward"] * steps
        if "left" in text:
            steps = NaVidBaseline._extract_repeats(text, divisor=30, default=1)
            return ["turn left"] * steps
        if "right" in text:
            steps = NaVidBaseline._extract_repeats(text, divisor=30, default=1)
            return ["turn right"] * steps
        return []

    @staticmethod
    def _extract_repeats(text: str, divisor: int, default: int) -> int:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match is None:
            return default
        value = abs(float(match.group()))
        return max(1, min(3, int(value / divisor)))

    @staticmethod
    def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            max_val = float(arr.max()) if arr.size else 1.0
            if max_val <= 1.01:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
