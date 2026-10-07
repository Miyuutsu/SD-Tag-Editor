from dataclasses import dataclass
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import orjson
import pandas as pd
import timm
import torch
import warnings

from huggingface_hub import hf_hub_download, constants
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image, UnidentifiedImageError
from simple_parsing import field, parse_known_args
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from tag_tree_functions import GroupTree, flatten_tags, load_groups, prune

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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
        try:
            img = Image.open(path)
            img = pil_ensure_rgb(img)
            img = pil_pad_square(img)
            tensor = self.transform(img)
            tensor = tensor[[2, 1, 0], ...]
            return tensor, str(path)
        except Exception as e:
            print(f"\n[WARNING] Skipping corrupt file: {path} - {e}")
            return None

def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), []
    return default_collate(batch)

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
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
    try:
        constants.HF_HUB_OFFLINE = True
        model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True).eval()

    except Exception:
        print(f"[INFO] {repo_id} not found in local cache.")
        print(f"Downloading model weights for {repo_id}...")
        constants.HF_HUB_OFFLINE = False
        model = timm.create_model(f"hf-hub:{repo_id}", pretrained=True).eval()

    finally:
        constants.HF_HUB_OFFLINE = False

    return model

def load_labels_hf(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision,
            token=token, local_files_only=True
        )
    except (LocalEntryNotFoundError, FileNotFoundError):
        try:
            print(f"Downloading tags for {repo_id}...")
            csv_path = hf_hub_download(
                repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
            )
        except HfHubHTTPError as e:
            raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    csv_path = Path(csv_path).resolve()
    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

def get_tags(
    probs: np.ndarray,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    prob_results = list(zip(labels.names, probs, strict=True))

    gen_labels = [prob_results[i] for i in labels.general]
    gen_labels = {x[0]: x[1] for x in gen_labels if x[1] > gen_threshold}
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    char_labels = [prob_results[i] for i in labels.character]
    char_labels = {x[0]: x[1] for x in char_labels if x[1] > char_threshold}
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    rating_labels = {prob_results[i][0]: prob_results[i][1] for i in labels.rating}

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
    tag_prefix: str = field(default="")
    dir_as_artist: bool = field(default=False)
    output_json: bool = field(default=False)


def prepare_inputs(model: str, image_or_images: str) -> tuple[str, Path]:
    try:
        repo_id = MODEL_REPO_MAP[model]
    except KeyError:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{model}'") from None
    image_path = Path(image_or_images.strip(' "'))
    if image_path.name == "NO_INPUT":
        image_path = Path(input("Input folder or image: ").strip(' "'))
    return repo_id, image_path

valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

def load_images(image_path: Path, subfolder: bool) -> list[Path]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image or Folder not found: {image_path}")
    if image_path.is_file():
        try:
            Image.open(image_path)
            return [image_path]
        except UnidentifiedImageError as e:
            raise UnidentifiedImageError(f"Unknown File Type of image: {image_path}") from e

    temp = image_path.rglob('*') if subfolder else image_path.iterdir()
    images: list[Path] = [
        x for x in temp
        if x.suffix.lower() in valid_extensions or x.suffix.lower() is None
    ]
    return images

def run_model(model: nn.Module, img_inputs: torch.Tensor) -> list[np.ndarray]:
    with torch.inference_mode():
        if torch_device.type != "cpu":
            img_inputs = img_inputs.to(
                torch_device,
                non_blocking=True
            )

        # 1. Raw Scores/Logits (Initial assignment)
        raw_outputs = model.forward(img_inputs)

        # 2. Convert to probabilities (Semantic transformation)
        probabilistic_outputs = F.sigmoid(raw_outputs)

        cpu_outputs = probabilistic_outputs.cpu().numpy()

    return list(cpu_outputs)


def setup(
    model_name: str, image_or_images: str, subfolder: bool, batch_size: int
):
    repo_id, image_path = prepare_inputs(model_name, image_or_images)
    image_paths = load_images(image_path, subfolder)
    model = load_model_hf(repo_id)

    if torch_device.type != "cpu":
        print("Moving model to GPU VRAM...")
        model = model.to(torch_device)

    print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    print("Creating data transform...")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    print("Creating Multi-Core DataLoader...")
    dataset = TaggerDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4,
        collate_fn=safe_collate
    )

    print("Loading prune tag groups...")
    group_tree: GroupTree = load_groups()

    return dataloader, model, labels, group_tree

class WorkerState:
    """Container to hold process-local state without triggering global warnings."""
    labels: LabelData
    group_tree: GroupTree
    opts: ScriptOptions

WORKER_STATE = WorkerState()

def init_worker(labels: LabelData, group_tree: GroupTree, opts: ScriptOptions):
    # Mutating attributes of an existing object does not require the 'global' keyword
    WORKER_STATE.labels = labels
    WORKER_STATE.group_tree = group_tree
    WORKER_STATE.opts = opts

def save_txt_output(args_tuple):
    """Worker function to process and save standard txt tags with prefixing logic."""
    img_tensor, path_str = args_tuple
    img_path = Path(path_str)
    real_path = img_path.resolve() # Resolves the symlink to find the true parent directory

    char_labels, gen_labels, _ = get_tags(
        probs=img_tensor, labels=WORKER_STATE.labels,
        gen_threshold=WORKER_STATE.opts.gen_threshold, char_threshold=WORKER_STATE.opts.char_threshold
    )
    pruned = flatten_tags(prune(WORKER_STATE.group_tree, {str(x): float(y) for x, y in gen_labels.items()}), True)

    # Format text
    pruned_formatted = [
        str(x[0]).replace("_", " ") if WORKER_STATE.opts.noUnderscores else str(x[0])
        for x in sorted(pruned, key=lambda x: x[1], reverse=True)
    ]
    if WORKER_STATE.opts.sortAlphabetically:
        pruned_formatted = sorted(pruned_formatted)

    final_tags = []

    # 1. Inject Human-Verified Directory Artist (Never shadow prefixed)
    if WORKER_STATE.opts.dir_as_artist:
        artist_name = real_path.parent.name.lower().replace(" ", "_")
        if WORKER_STATE.opts.noUnderscores:
            artist_name = artist_name.replace("_", " ")
        final_tags.append(f"artist:{artist_name}")

    # 2. Append AI Characters
    for c in char_labels:
        char_tag = c.replace("_", " ") if WORKER_STATE.opts.noUnderscores else c
        final_tags.append(f"{WORKER_STATE.opts.tag_prefix}{char_tag}")

    # 3. Append AI General Tags
    for g in pruned_formatted:
        final_tags.append(f"{WORKER_STATE.opts.tag_prefix}{g}")

    pruned_str = ", ".join(final_tags)
    img_path.with_suffix(".txt").write_text(pruned_str, encoding="utf-8")

def save_json_output(args_tuple):
    img_tensor, path_str = args_tuple
    js = Path(path_str).with_suffix(".json")
    real_path = Path(path_str).resolve()

    char, gen, rating = get_tags(
        probs=img_tensor, labels=WORKER_STATE.labels,
        gen_threshold=WORKER_STATE.opts.gen_threshold, char_threshold=WORKER_STATE.opts.char_threshold
    )
    artist = []

    pruned_gen_tuples = flatten_tags(prune(WORKER_STATE.group_tree, gen), True)

    # Apply Optional Prefixes
    prefix = WORKER_STATE.opts.tag_prefix
    char = {f"{prefix}{str(k)}": float(v) for k, v in char.items()}
    gen = {f"{prefix}{str(k)}": float(v) for k, v in pruned_gen_tuples}
    rating = {str(k): float(v) for k, v in rating.items()}

    # Append the un-prefixed artist tag
    if WORKER_STATE.opts.dir_as_artist:
        artist_name = real_path.parent.name.lower().replace(" ", "_")
        artist.append(artist_name)

    if js.is_file():
        data = orjson.loads(js.read_bytes())
        for x in data.get("character", {}):
            char[x] = (char.get(x, WORKER_STATE.opts.char_threshold) + WORKER_STATE.opts.char_threshold) / 2
        for x in data.get("general", {}):
            gen[x] = (gen.get(x, WORKER_STATE.opts.gen_threshold) + WORKER_STATE.opts.gen_threshold) / 2

        existing_artist = data.get("artist", [])
        for a in existing_artist:
            if a not in artist:
                artist.append(a)

    output_data = {
        "character": char,
        "general": gen,
        "rating": rating,
        "artist": artist
    }
    js.write_bytes(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))

def main(opts: ScriptOptions):
    dataloader, model, labels, group_tree = setup(
        opts.model, opts.image_or_images, opts.subfolder, opts.batch_size
    )

    target_worker = save_json_output if opts.output_json else save_txt_output
    futures = []

    with ProcessPoolExecutor(
        max_workers=14,
        initializer=init_worker,
        initargs=(labels, group_tree, opts)
    ) as executor:

        for img_inputs, paths in tqdm(dataloader):
            if len(paths) == 0:
                continue

            outputs = run_model(model, img_inputs)
            for task in zip(outputs, paths, strict=True):
                futures.append(executor.submit(target_worker, task))

        for future in tqdm(futures, desc="Saving files"):
            future.result()

if __name__ == "__main__":
    parsed_opts, _ = parse_known_args(ScriptOptions)
    assert isinstance(parsed_opts, ScriptOptions)
    main(parsed_opts)
