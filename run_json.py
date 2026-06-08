# pylint: disable=duplicate-code,line-too-long,too-many-locals,too-many-branches,too-many-statements,too-many-arguments,too-many-positional-arguments,missing-function-docstring,missing-module-docstring,missing-class-docstring,no-member,c-extension-no-member
from dataclasses import dataclass, field
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
import orjson
from simple_parsing import parse_known_args
from tqdm import tqdm

from run import MODEL_REPO_MAP, ScriptOptions, get_tags, run_model, setup
from tag_tree_functions import GroupTree, flatten_tags, prune

WORKER_LABELS = None
WORKER_GROUP_TREE = None
WORKER_OPTS = None

def init_worker(labels, group_tree, opts):
    global WORKER_LABELS, WORKER_GROUP_TREE, WORKER_OPTS # pylint: disable=global-statement
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
    real_path = Path(path_str).resolve()

    char, gen, rating = get_tags(
        probs=img_tensor, labels=WORKER_LABELS,
        gen_threshold=WORKER_OPTS.gen_threshold, char_threshold=WORKER_OPTS.char_threshold
    )
    artist = []

    pruned_gen_tuples = flatten_tags(prune(WORKER_GROUP_TREE, gen), True)

    # Apply Optional Prefixes
    prefix = WORKER_OPTS.tag_prefix
    char = {f"{prefix}{str(k)}": float(v) for k, v in char.items()}
    gen = {f"{prefix}{str(k)}": float(v) for k, v in pruned_gen_tuples}
    rating = {str(k): float(v) for k, v in rating.items()}

    # Append the un-prefixed artist tag
    if WORKER_OPTS.dir_as_artist:
        artist_name = real_path.parent.name.lower().replace(" ", "_")
        artist.append(artist_name)

    if js.is_file():
        data = orjson.loads(js.read_bytes())
        for x in data.get("character", {}):
            char[x] = (char.get(x, WORKER_OPTS.char_threshold) + WORKER_OPTS.char_threshold) / 2
        for x in data.get("general", {}):
            gen[x] = (gen.get(x, WORKER_OPTS.gen_threshold) + WORKER_OPTS.gen_threshold) / 2

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

    init_worker(labels, group_tree, opts)

    with ProcessPoolExecutor(
        max_workers=14,
        initializer=init_worker,
        initargs=(labels, group_tree, opts)
    ) as executor:

        for img_inputs, paths in tqdm(dataloader):
            if len(paths) == 0:
                continue
            outputs = run_model(model, img_inputs)
            tasks = [(img, paths[i]) for i, img in enumerate(outputs)]

            list(executor.map(save_json_output, tasks))

if __name__ == "__main__":
    parsed_opts, _ = parse_known_args(ScriptOptions)
    if parsed_opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{parsed_opts.model}'")
    main(parsed_opts)
