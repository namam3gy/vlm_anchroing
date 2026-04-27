from __future__ import annotations

import argparse
import io
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


DEFAULT_BACKEND = "diffusers"
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_OUTPUT_DIR = Path("inputs/irrelevant_number")
DEFAULT_NEGATIVE_PROMPT = (
    "extra numbers, extra text, letters, words, watermark, logo, signature, "
    "blurry digits, cropped digits, tiny number, hidden number, low contrast, duplicate numerals"
)
HF_API_URL = "https://router.huggingface.co/models/{model}"


@dataclass(frozen=True)
class SceneTemplate:
    name: str
    description: str


@dataclass(frozen=True)
class RenderConfig:
    backend: str
    model: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    max_sequence_length: int
    device: str | None = None
    dtype: str | None = None
    local_files_only: bool = False


SCENE_TEMPLATES: list[SceneTemplate] = [
    SceneTemplate(
        name="speed_limit_sign",
        description=(
            "a realistic roadside speed-limit sign with a bold black clear numeral {number} inside a red circle, "
            "close-up, the sign fills most of the frame"
        ),
    ),
    SceneTemplate(
        name="championship_belt",
        description=(
            "a premium championship belt on dark velvet with a large engraved clear numeral {number} in the center plate, "
            "dramatic studio lighting, centered composition"
        ),
    ),
    SceneTemplate(
        name="sports_jersey",
        description=(
            "a professional sports jersey hanging in a locker room with the large clear numeral {number} on the chest, "
            "front-facing, crisp fabric detail"
        ),
    ),
    SceneTemplate(
        name="house_plaque",
        description=(
            "an elegant brass house-number plaque by a front door, showing the clear numeral {number}, "
            "tight composition, shallow depth of field"
        ),
    ),
    SceneTemplate(
        name="race_car",
        description=(
            "a race car door in a pit lane with the clear numeral {number} painted very large on the side, "
            "dynamic but readable, realistic motorsport scene"
        ),
    ),
    SceneTemplate(
        name="scoreboard",
        description=(
            "a stadium scoreboard at dusk dominated by the glowing clear numeral {number}, "
            "centered, realistic stadium lighting"
        ),
    ),
    SceneTemplate(
        name="marathon_bib",
        description=(
            "a marathon race bib pinned to a running shirt, displaying the clear numeral {number} very large, "
            "clean close-up, realistic fabric and paper texture"
        ),
    ),
    SceneTemplate(
        name="hotel_door",
        description=(
            "a luxury hotel room door with a metal plaque that shows the clear numeral {number}, "
            "centered, cinematic hallway lighting"
        ),
    ),
    SceneTemplate(
        name="parking_marker",
        description=(
            "a parking garage pillar marker showing the clear numeral {number} in huge print, "
            "industrial concrete background, close-up"
        ),
    ),
    SceneTemplate(
        name="subway_tile",
        description=(
            "a subway platform wall tile with the clear numeral {number} shown prominently, "
            "clean centered framing, realistic station materials"
        ),
    ),
    SceneTemplate(
        name="locker_badge",
        description=(
            "a metal locker badge with the clear numeral {number} embossed large and centered, "
            "close-up, realistic reflections"
        ),
    ),
]

STYLE_MODIFIERS = [
    "photorealistic, high detail, natural lighting",
    "realistic editorial photo, sharp focus, premium materials",
    "commercial photography, highly legible subject, clean composition",
    "cinematic realism, believable textures, soft background separation",
]

COLOR_MODIFIERS = [
    "strong red, white, and black contrast",
    "gold and black luxury tones",
    "blue and white sports colors",
    "orange, charcoal, and silver industrial tones",
    "green and cream outdoor tones",
    "navy, brass, and warm ambient light",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate irrelevant number images 1..100 and save them as "
            "inputs/irrelevant_number/<number>.png. By default this uses a local "
            "Diffusers text-to-image model downloaded from Hugging Face."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("diffusers", "hf-api"),
        default=DEFAULT_BACKEND,
        help="Image generation backend. Defaults to a local Diffusers pipeline.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model id. Default is a local SDXL Turbo checkpoint for Diffusers; "
            "use a different text-to-image repo id if desired."
        ),
    )
    parser.add_argument("--start", type=int, default=1, help="First number to generate.")
    parser.add_argument("--end", type=int, default=100, help="Last number to generate.")
    parser.add_argument(
        "--numbers",
        default=None,
        help=(
            "Comma-separated explicit list of integers to generate (overrides "
            "--start/--end). Example: '9,10,15,20,100,200'."
        ),
    )
    parser.add_argument("--width", type=int, default=None, help="Output width. Defaults depend on the model family.")
    parser.add_argument("--height", type=int, default=None, help="Output height. Defaults depend on the model family.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps. Defaults depend on the backend/model family.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale. Defaults depend on the backend/model family.",
    )
    parser.add_argument("--seed-base", type=int, default=1729, help="Per-image seed is seed_base + number.")
    parser.add_argument(
        "--scene-offset",
        type=int,
        default=0,
        help=(
            "Shift scene-template selection by this amount (mod len(SCENE_TEMPLATES)). "
            "Useful when retrying a stubborn number on a different scene."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pause between generated images.")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without loading a model.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for the local Diffusers backend.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype for the local Diffusers backend.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download model files; only use files already cached locally.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Only used for FLUX-family Diffusers pipelines.",
    )
    return parser.parse_args()


def resolve_hf_token(required: bool) -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if required and not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN before running this script.")
    return token


def normalize_model_family(model: str) -> str:
    lower_model = model.lower()
    if "flux" in lower_model:
        return "flux"
    if "sdxl-turbo" in lower_model:
        return "sdxl_turbo"
    if "sd-turbo" in lower_model:
        return "sd_turbo"
    return "generic"


def resolve_width_height(width: int | None, height: int | None, model_family: str) -> tuple[int, int]:
    if width is not None and height is not None:
        return width, height

    if model_family == "sd_turbo":
        default_width, default_height = 512, 512
    else:
        default_width, default_height = 480, 480

    return width or default_width, height or default_height


def resolve_steps_guidance(
    backend: str,
    model_family: str,
    steps: int | None,
    guidance_scale: float | None,
) -> tuple[int, float]:
    if steps is not None and guidance_scale is not None:
        return steps, guidance_scale

    if backend == "hf-api":
        default_steps, default_guidance = 28, 4.0
    elif model_family == "flux":
        default_steps, default_guidance = 4, 0.0
    elif model_family in {"sdxl_turbo", "sd_turbo"}:
        default_steps, default_guidance = 1, 0.0
    else:
        default_steps, default_guidance = 28, 4.0

    return steps or default_steps, guidance_scale if guidance_scale is not None else default_guidance


def build_render_config(args: argparse.Namespace) -> RenderConfig:
    model_family = normalize_model_family(args.model)
    width, height = resolve_width_height(args.width, args.height, model_family)
    steps, guidance_scale = resolve_steps_guidance(
        backend=args.backend,
        model_family=model_family,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    device = None if args.backend == "hf-api" else args.device
    dtype = None if args.backend == "hf-api" else args.dtype
    return RenderConfig(
        backend=args.backend,
        model=args.model,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        max_sequence_length=args.max_sequence_length,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )


def build_prompt(number: int, seed_base: int, scene_offset: int = 0) -> str:
    rng = random.Random(seed_base + number)
    scene = SCENE_TEMPLATES[(number - 1 + scene_offset) % len(SCENE_TEMPLATES)]
    style = rng.choice(STYLE_MODIFIERS)
    colors = rng.choice(COLOR_MODIFIERS)
    return (
        f"{style}. {colors}. "
        f"{scene.description.format(number=number)}. "
        f"Only numeral {number}. Large and easy to read. "
        "Photorealistic. No text. No duplicate numerals. Generate just single centered numeral with no extra digits."
    )


def request_image_bytes(
    session: requests.Session,
    model: str,
    token: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    timeout_seconds: float,
    max_retries: int,
) -> bytes:
    url = HF_API_URL.format(model=model)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        },
        "options": {"wait_for_model": True},
    }

    last_error: RuntimeError | None = None
    for attempt in range(1, max_retries + 1):
        response = session.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        content_type = response.headers.get("content-type", "")

        if response.ok and content_type.startswith("image/"):
            return response.content

        if "application/json" in content_type:
            details = response.json()
            if response.status_code == 503:
                wait_seconds = float(details.get("estimated_time", min(20, attempt * 3)))
                time.sleep(wait_seconds)
                continue
            error_message = details.get("error", details)
        else:
            error_message = response.text.strip() or f"HTTP {response.status_code}"

        last_error = RuntimeError(
            f"inference request failed on attempt {attempt}/{max_retries}: {error_message}"
        )
        if attempt < max_retries:
            time.sleep(min(20, attempt * 3))

    raise last_error or RuntimeError("inference request failed without an error payload")


def resolve_local_device(requested_device: str, torch) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(requested_dtype: str, model_family: str, device: str, torch):
    if requested_dtype == "float16":
        return torch.float16
    if requested_dtype == "bfloat16":
        return torch.bfloat16
    if requested_dtype == "float32":
        return torch.float32

    if device == "cpu":
        return torch.float32

    if model_family == "sdxl_turbo":
        return torch.float16
    if model_family == "sd_turbo":
        return torch.float16
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class DiffusersRenderer:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config
        self.model_family = normalize_model_family(config.model)

        try:
            import torch
            from diffusers import AutoPipelineForText2Image, FluxPipeline
        except ImportError as exc:
            raise SystemExit(
                "The local Diffusers backend requires diffusers to be installed. "
                "Run `uv sync` or `pip install -e .` to install the updated dependencies."
            ) from exc

        self._torch = torch
        device = resolve_local_device(config.device or "auto", torch)
        self.device = device
        self.torch_dtype = resolve_torch_dtype(config.dtype or "auto", self.model_family, device, torch)

        from_pretrained_kwargs = {
            "torch_dtype": self.torch_dtype,
            "local_files_only": config.local_files_only,
        }

        if self.model_family == "flux":
            loader = FluxPipeline.from_pretrained
        else:
            if self.model_family == "sdxl_turbo" and self.torch_dtype == torch.float16:
                from_pretrained_kwargs["variant"] = "fp16"
            loader = AutoPipelineForText2Image.from_pretrained

        try:
            self.pipeline = loader(config.model, **from_pretrained_kwargs)
        except Exception as exc:
            if "gated repo" in str(exc).lower() or "403" in str(exc):
                raise SystemExit(
                    "The selected model is gated on Hugging Face. Accept its access terms first, "
                    "or use an open model such as `stabilityai/sdxl-turbo`."
                ) from exc
            raise

        self.pipeline = self.pipeline.to(device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)

    def render(self, prompt: str, negative_prompt: str, seed: int) -> bytes:
        call_kwargs = {
            "prompt": prompt,
            "width": self.config.width,
            "height": self.config.height,
            "num_inference_steps": self.config.steps,
            "guidance_scale": self.config.guidance_scale,
            "generator": self._torch.Generator(device="cpu").manual_seed(seed),
        }
        if self.model_family == "flux":
            call_kwargs["max_sequence_length"] = self.config.max_sequence_length
        else:
            call_kwargs["negative_prompt"] = negative_prompt

        image = self.pipeline(**call_kwargs).images[0]
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        return image_bytes.getvalue()


def save_png(image_bytes: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.convert("RGB").save(target, format="PNG")


def main() -> None:
    args = parse_args()
    if args.numbers is not None:
        try:
            numbers = sorted({int(x) for x in args.numbers.split(",") if x.strip()})
        except ValueError as exc:
            raise SystemExit(f"--numbers must be a comma-separated list of integers: {exc}")
        if not numbers:
            raise SystemExit("--numbers must contain at least one integer.")
    else:
        if args.start < 0 or args.end < args.start:
            raise SystemExit("--start and --end must define a valid inclusive range with start >= 1.")
        numbers = list(range(args.start, args.end + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_config = build_render_config(args)
    token = ""
    if not args.dry_run and render_config.backend == "hf-api":
        token = resolve_hf_token(required=True) or ""

    session: requests.Session | None = None
    diffusers_renderer: DiffusersRenderer | None = None
    if not args.dry_run and render_config.backend == "hf-api":
        session = requests.Session()
    if not args.dry_run and render_config.backend == "diffusers":
        diffusers_renderer = DiffusersRenderer(render_config)

    try:
        for number in tqdm(numbers, desc="Generating number images"):
            target = args.output_dir / f"{number}.png"
            prompt = build_prompt(
                number=number, seed_base=args.seed_base, scene_offset=args.scene_offset
            )

            if args.dry_run:
                print(f"{number}: {prompt}")
                continue

            if target.exists() and not args.overwrite:
                continue

            if render_config.backend == "diffusers":
                if diffusers_renderer is None:
                    raise RuntimeError("Diffusers backend renderer was not initialized.")
                image_bytes = diffusers_renderer.render(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    seed=args.seed_base + number,
                )
            else:
                if session is None:
                    raise RuntimeError("HF API backend session was not initialized.")
                image_bytes = request_image_bytes(
                    session=session,
                    model=render_config.model,
                    token=token,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    width=render_config.width,
                    height=render_config.height,
                    steps=render_config.steps,
                    guidance_scale=render_config.guidance_scale,
                    seed=args.seed_base + number,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                )

            save_png(image_bytes=image_bytes, target=target)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()
