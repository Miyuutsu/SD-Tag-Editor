import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import timm
import torch
import tqdm
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image, UnidentifiedImageError
from simple_parsing import field, parse_known_args
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tag_tree_functions import GroupTree, flatten_tags, load_groups, prune

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "eva02": "SmilingWolf/wd-eva02-large-tagger-v3",
}

class TaggerDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        img = pil_ensure_rgb(img)
        img = pil_pad_square(img)
        tensor = self.transform(img)
        tensor = tensor[[2, 1, 0], :, :]  # BGR swap
        return tensor, str(path)

def list_files(path: Path) -> list[Path]:
    folders = [path]
    files = []

    while folders:
        folder = folders.pop(0)
        for file in folder.iterdir():
            if file.is_dir():
                folders.append(file)
                continue
            files.append(file)
    return files


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_model_hf(repo_id: str) -> nn.Module:
    model: nn.Module = timm.create_model(f"hf-hub:{repo_id}").eval()
    state_dict = timm.models.load_state_dict_from_hf(repo_id)
    model.load_state_dict(state_dict)
    return model


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token)
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )


def _process_single_image(img_path: Path, transform) -> torch.Tensor:
    """Helper function to process a single image start-to-finish."""
    img = Image.open(img_path)
    img = pil_ensure_rgb(img)
    img = pil_pad_square(img)
    tensor = transform(img).unsqueeze(0)
    tensor = tensor[:, [2, 1, 0]]  # BGR swap
    return tensor


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs.numpy()))

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # rating labels
    rating_labels = dict([probs[i] for i in labels.rating])

    return char_labels, gen_labels, rating_labels


@dataclass
class ScriptOptions:
    image_or_images: str = field(positional=True, default="NO_INPUT")
    batch_size: int = field(default=1)
    model: str = field(default="vit-large")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)
    subfolder: bool = field(default=False)
    noUnderscores: bool = field(default=True)
    sortAlphabetically: bool = field(default=False)


def prepare_inputs(model: str, image_or_images: str) -> tuple[str | None, Path | None]:
    repo_id = MODEL_REPO_MAP.get(model)
    image_path = Path(image_or_images.strip(' "'))
    if image_path.name == "NO_INPUT":
        image_path = Path(input("Input folder or image: ").strip(' "'))
    return repo_id, image_path


def load_images(image_path: Path, subfolder: bool) -> list[Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image or Folder not found: {image_path}")
    if image_path.is_file():
        try:
            Image.open(image_path)
            return [image_path]
        except UnidentifiedImageError as e:
            raise UnidentifiedImageError(f"Unknown File Type of image: {image_path}") from e

    images: list[Path] = []
    if subfolder:
        temp = list_files(image_path)
    else:
        temp = [x for x in image_path.iterdir() if x.is_file()]

    images: list[Path] = [file for file in temp]
    return images

def run_model(model: nn.Module, img_inputs: torch.Tensor) -> list[torch.Tensor]:
    """Runs inference without constantly reloading the model to CPU."""
    with torch.inference_mode():
        if torch.device.type != "cpu":
            img_inputs = img_inputs.to(torch_device)
        outputs = model.forward(img_inputs)
        outputs = F.sigmoid(outputs)
        if torch.device.type != "cpu":
            outputs = outputs.to("cpu")
            # We explicitly removed model.to("cpu") here!
        outputs = torch.unbind(outputs, dim=0)
    return outputs

def setup(
    model_name: str, image_or_images: str, subfolder: bool, batch_size: int
):
    repo_id, image_path = prepare_inputs(model_name, image_or_images)
    image_paths = load_images(image_path, subfolder)

    print(f"Loading model '{model_name}' from '{repo_id}'...")
    model = load_model_hf(repo_id)
    if torch.device.type != "cpu":
        print("Moving model to GPU VRAM...")
        model = model.to(torch_device) # Moved ONCE!

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Creating data transform...")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    print("Creating Multi-Core DataLoader...")
    dataset = TaggerDataset(image_paths, transform)
    # Using 8 CPU workers bypasses the GIL and continuously feeds the 4080
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,      # Locks memory for instant PCIe transfer
        prefetch_factor=4     # Forces CPU to aggressively queue up batches
    )

    print("Loading prune tag groups...")
    group_tree: GroupTree = load_groups()

    # Notice we return dataloader instead of batches, and we no longer need to pass transform
    return dataloader, model, labels, group_tree


# Global placeholders for the worker processes
WORKER_LABELS = None
WORKER_GROUP_TREE = None
WORKER_OPTS = None

def init_worker(labels, group_tree, opts):
    """Initializes the heavy static data ONCE per background worker."""
    global WORKER_LABELS, WORKER_GROUP_TREE, WORKER_OPTS
    WORKER_LABELS = labels
    WORKER_GROUP_TREE = group_tree
    WORKER_OPTS = opts

def save_txt_output(args_tuple):
    """Worker function to process and save standard txt tags."""
    img_tensor, path_str = args_tuple

    char_labels, gen_labels, _ = get_tags(
        probs=img_tensor, labels=WORKER_LABELS,
        gen_threshold=WORKER_OPTS.gen_threshold, char_threshold=WORKER_OPTS.char_threshold
    )
    pruned = flatten_tags(prune(WORKER_GROUP_TREE, {str(x): float(y) for x, y in gen_labels.items()}), True)
    pruned = [
        x[0].replace("_", " ") if WORKER_OPTS.noUnderscores else x[0]
        for x in sorted(pruned, key=lambda x: x[1], reverse=True)
    ]
    if WORKER_OPTS.sortAlphabetically:
        pruned = sorted(pruned)
    pruned = ", ".join([str(x) for x in char_labels] + pruned)
    Path(path_str).with_suffix(".txt").write_text(pruned, encoding="utf-8")


def main(opts: ScriptOptions):
    dataloader, model, labels, group_tree = setup(
        opts.model, opts.image_or_images, opts.subfolder, opts.batch_size
    )

    # Initialize globals locally since threads share the memory space
    init_worker(labels, group_tree, opts)

    # Switch to ThreadPool to prevent tensor IPC serialization locks
    with ThreadPoolExecutor(max_workers=14) as executor:
        for img_inputs, paths in tqdm(dataloader):
            outputs = run_model(model, img_inputs)
            tasks = [(img, paths[i]) for i, img in enumerate(outputs)]
            list(executor.map(save_json_output, tasks))

if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
