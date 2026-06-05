from dataclasses import dataclass, field
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor # Change this import
import orjson
from simple_parsing import parse_known_args
from tqdm import tqdm

from run import MODEL_REPO_MAP, ScriptOptions, get_tags, run_model, setup
from tag_tree_functions import GroupTree, flatten_tags, prune

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

@dataclass
class Item:
    character: list[str] = field(default_factory=lambda: [])
    general: list[str] = field(default_factory=lambda: [])
    artist: list[str] = field(default_factory=lambda: [])
    rating: dict[str, float] = field(default_factory=lambda: {})

def process_tags(
    group_tree: GroupTree,
    char_labels: dict[str, float],
    gen_labels: dict[str, float],
    ratings: dict[str, float],
    artists: list[str] = None,
) -> Item:
    if artists is None:
        artists = []
    char_labels = list(char_labels.keys())
    gen_labels = flatten_tags(prune(group_tree, dict(gen_labels)), True)
    gen_labels = [x[0] for x in sorted(gen_labels, key=lambda x: x[1], reverse=True)]
    return Item(
        character=char_labels,
        general=gen_labels,
        artist=artists,
        rating={str(x): float(y) for x, y in ratings.items()},
    )

def save_json_output(args_tuple):
    img_tensor, path_str = args_tuple
    js = Path(path_str).with_suffix(".json")

    char, gen, rating = get_tags(
        probs=img_tensor, labels=WORKER_LABELS,
        gen_threshold=WORKER_OPTS.gen_threshold, char_threshold=WORKER_OPTS.char_threshold
    )
    artist = []

    # FIX 1: Prune general tags to avoid database bloat
    pruned_gen_tuples = flatten_tags(prune(WORKER_GROUP_TREE, gen), True)

    # FIX 2: Cast all PyTorch numpy.float32 scalars to native Python floats
    char = {str(k): float(v) for k, v in char.items()}
    gen = {str(k): float(v) for k, v in pruned_gen_tuples}
    rating = {str(k): float(v) for k, v in rating.items()}

    if js.is_file():
        data = orjson.loads(js.read_bytes())
        for x in data.get("character", {}):
            char[x] = (char.get(x, WORKER_OPTS.char_threshold) + WORKER_OPTS.char_threshold) / 2
        for x in data.get("general", {}):
            gen[x] = (gen.get(x, WORKER_OPTS.gen_threshold) + WORKER_OPTS.gen_threshold) / 2
        artist = data.get("artist", [])

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

    # Initialize the globals in the main thread since Threads share memory space
    init_worker(labels, group_tree, opts)

    # FIX: Use ThreadPoolExecutor to prevent deadlocks with PyTorch's DataLoader processes
    with ProcessPoolExecutor(
    max_workers=14,
    initializer=init_worker,
    initargs=(labels, group_tree, opts)
    ) as executor:

        for img_inputs, paths in tqdm(dataloader): # tqdm is imported as tqdm in run_json, tqdm.tqdm in run
            outputs = run_model(model, img_inputs)
            tasks = [(img, paths[i]) for i, img in enumerate(outputs)]

            # Threads share memory, eliminating PyTorch tensor pickling crashes
            list(executor.map(save_json_output, tasks)) # Use save_txt_output for run.py

if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
