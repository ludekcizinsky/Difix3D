import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tyro
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline_difix import DifixPipeline


def save_labeled_comparison(
    images,
    labels,
    output_path,
    label_height=40,
    background_color="black",
    text_color="white",
    font_path=None,
    font_size=18,
):
    """Stack images horizontally and add centered labels above each column."""
    if len(images) != len(labels):
        raise ValueError("Number of images must match number of labels")

    widths = [img.width for img in images]
    heights = [img.height for img in images]

    total_width = sum(widths)
    max_height = max(heights)

    label_band = max(label_height, font_size + 10)
    combined_image = Image.new("RGB", (total_width, max_height + label_band), background_color)
    draw = ImageDraw.Draw(combined_image)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    x_offset = 0
    for width, height, label, image in zip(widths, heights, labels, images):
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x_offset + (width - text_width) // 2
        text_y = (label_band - text_height) // 2
        draw.text((text_x, text_y), label, fill=text_color, font=font)

        y_offset = label_band + (max_height - height) // 2
        combined_image.paste(image, (x_offset, y_offset))
        x_offset += width

    combined_image.save(output_path)


@dataclass
class DemoConfig:
    """CLI configuration for running the DiFix demo."""

    model_id: str = "nvidia/difix_ref"
    root_dir: Path = Path(
        "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance/lhm/evaluation/difix/epoch_0000"
    )
    src_cam_id: int = 28
    tgt_cam_id: int = 4
    prompt: str = "remove degradation"
    output_dir: Path = Path("/home/cizinsky/difix3d/playground/outputs/")
    num_images: int = 125
    num_inference_steps: int = 1
    timesteps: Tuple[int, ...] = (199,)
    guidance_scale: float = 0.0
    torch_home: Path = Path("/scratch/izar/cizinsky/.cache")
    hf_home: Path = Path("/scratch/izar/cizinsky/.cache")
    device: str = "cuda"

def run_demo(config: DemoConfig) -> None:
    os.environ["TORCH_HOME"] = str(config.torch_home)
    os.environ["HF_HOME"] = str(config.hf_home)

    pipe = DifixPipeline.from_pretrained(config.model_id, trust_remote_code=True)
    pipe.to(config.device)
    pipe.set_progress_bar_config(disable=True)

    if (config.output_dir / f"refined_cam{config.tgt_cam_id}").exists():
        # remove existing directory to avoid mixing old and new results
        shutil.rmtree(config.output_dir / f"refined_cam{config.tgt_cam_id}")

    config.output_dir = config.output_dir / f"refined_cam{config.tgt_cam_id}" / "frames"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(config.num_images), desc="Processing images"):
        frame_name = f"{i + 1:06d}.jpg"

        # Load image to refine
        to_refine_img_path = (
            config.root_dir / f"{config.tgt_cam_id}" / "difix_inputs" / "renders" / frame_name
        )
        if not os.path.exists(to_refine_img_path):
            raise FileNotFoundError(f"Image to refine not found at {to_refine_img_path}")
        to_refine_image = load_image(str(to_refine_img_path))

        # Load reference image
        ref_img_path = (
            config.root_dir / f"{config.src_cam_id}" / "difix_inputs" / "gt_frames" / frame_name
        )
        if not os.path.exists(ref_img_path):
            raise FileNotFoundError(f"Reference image not found at {ref_img_path}")
        ref_image = load_image(str(ref_img_path))

        # Generate output image (refinement)
        output_image = pipe(
            config.prompt,
            image=to_refine_image,
            ref_image=ref_image,
            num_inference_steps=config.num_inference_steps,
            timesteps=list(config.timesteps),
            guidance_scale=config.guidance_scale,
        ).images[0]

        # Load gt image for comparison
        gt_img_path = (
            config.root_dir / f"{config.tgt_cam_id}" / "difix_inputs" / "gt_frames" / frame_name
        )
        if not os.path.exists(gt_img_path):
            raise FileNotFoundError(f"Gt image not found at {gt_img_path}")
        gt_image = load_image(str(gt_img_path))

        # Save comparison image
        save_labeled_comparison(
            images=[ref_image, to_refine_image, output_image, gt_image],
            labels=["Reference", "To refine", "Output", "Ground truth"],
            output_path=config.output_dir / f"{i + 1:06d}.png",
        )


if __name__ == "__main__":
    run_demo(tyro.cli(DemoConfig))
