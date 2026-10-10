"""Fixed check prompts. Scenes are generated in the job. No photos of real people."""

from __future__ import annotations

BASE_PROMPTS: tuple[str, ...] = (
    "A quiet coastal cliff at dawn, pale water and layered rock, no people.",
    "A red desert mesa under a high white cloud, late afternoon light.",
    "Birch trees along a still lake, mist close to the water.",
    "A snowy pine ridge with a single empty footpath.",
    "Terraced rice fields in soft rain, no workers in the frame.",
    "A ceramic teapot and two cups on a wooden table, window light.",
    "A brass desk lamp, a closed book, and a pencil on dark oak.",
    "A bicycle leaning against a sunlit brick wall.",
    "A bowl of oranges on a linen cloth in a simple kitchen.",
    "A row of glass bottles on a workshop shelf, dusty light.",
    "A stone footbridge over a narrow canal, empty of people.",
    "A concrete concert hall exterior, overcast sky, wide steps.",
    "A small library reading room with empty oak chairs.",
    "A greenhouse of ferns, wet glass, no visitors.",
    "A lighthouse on a grass headland, choppy grey sea.",
    "Closed night-market stalls, paper lanterns, wet pavement, no crowd.",
    "An adult woman in a wool coat on a station platform, fully clothed, looking at the clock.",
    "An adult man in a linen shirt and trousers reading a newspaper on a park bench.",
    "An adult painter in a denim apron standing beside a blank canvas in a studio.",
    "Two adults in winter coats walking away along a tree-lined avenue.",
    "An adult chef in a white jacket plating vegetables in a quiet kitchen.",
    "An adult architect in a tailored jacket reviewing drawings at a standing desk.",
    "A sunlit apartment balcony with potted herbs and a folding chair.",
    "A marble staircase in a museum, rope barrier, no visitors.",
    "A mountain hut with a smoking chimney, evening, empty yard.",
    "A field of sunflowers under a blue sky, no people.",
    "A close view of a mechanical wristwatch on a cloth.",
    "A red tram on an empty city street after rain.",
    "A sandstone courtyard with a single olive tree.",
    "An adult woman in a long coat and scarf crossing a quiet square, fully clothed.",
    "A stack of folded wool blankets on a window seat.",
    "A modern chapel interior, concrete and wood, empty pews.",
)

CONTROL_PROMPTS: tuple[str, ...] = (
    "An adult hiker in a red jacket and trousers on a ridge path, fully clothed, facing away.",
    "A wooden chair and a round table in an empty room.",
    "An adult dancer in a long coat and boots holding a still pose in a studio, fully clothed.",
    "A bicycle beside a lamp post on an empty street.",
    "An adult gardener in a canvas jacket kneeling by potted shrubs, fully clothed.",
    "A stone arch bridge over a river, no boats and no people.",
    "An adult courier with a backpack standing at a crosswalk, fully clothed.",
    "A vase of branches on a pedestal in a gallery.",
    "An adult musician seated with a cello, concert clothes, empty hall.",
    "A greenhouse aisle with a wheelbarrow.",
    "An adult swimmer in a full tracksuit standing on a pool deck, not in the water.",
    "A lighthouse stair and railing, no people.",
    "An adult teacher in a cardigan at a blackboard, empty classroom.",
    "A park bench under a plane tree, empty.",
    "An adult photographer in a field jacket standing on a dune, fully clothed.",
    "A kitchen counter with a cutting board and a kettle.",
)

INPAINT_PROMPTS: tuple[str, ...] = (
    "Fill the missing patch with the same oak tabletop.",
    "Fill the missing patch with a plain ceramic cup.",
    "Fill the missing patch with the continuation of the brick wall.",
    "Fill the missing patch with a folded linen napkin.",
    "Fill the missing patch with the same cloudy sky.",
    "Fill the missing patch with a closed hardcover book.",
    "Fill the missing patch with matching floor tiles.",
    "Fill the missing patch with a small potted fern.",
)

# (instruction, scene seed). The source image is rendered in the job from the seed.
EDIT_PAIRS: tuple[tuple[str, int], ...] = (
    ("Make the sky overcast.", 0),
    ("Change the coat to a navy wool coat.", 1),
    ("Turn the wooden table into a dark metal table.", 2),
    ("Add a soft morning haze.", 3),
    ("Change the jacket to a grey raincoat.", 4),
    ("Make the leaves yellow as in autumn.", 5),
    ("Replace the paper lanterns with simple globe lamps.", 6),
    ("Change the scarf to a deep green scarf.", 7),
    ("Make the sea calmer and greyer.", 8),
    ("Turn the daylight into late afternoon sun.", 9),
    ("Change the apron to a black apron.", 10),
    ("Add a thin layer of snow on the roof only.", 11),
    ("Make the bicycle red.", 12),
    ("Change the curtains to heavy linen.", 13),
    ("Turn the grass a drier summer yellow.", 14),
    ("Change the boots to brown leather boots.", 15),
    ("Make the room lighting warmer.", 16),
    ("Replace the teapot with a glass carafe.", 17),
    ("Add low fog around the bridge piers.", 18),
    ("Change the shirt to a pale blue shirt.", 19),
    ("Make the path wet from rain.", 20),
    ("Turn the concrete wall a warmer stone colour.", 21),
    ("Change the hat to a wool cap.", 22),
    ("Make the background a quiet library, keeping the clothed adult.", 23),
)

BASE_SEEDS: tuple[int, ...] = (0, 1)
CONTROL_SEEDS: tuple[int, ...] = (0, 1)
CONTROL_TYPES: tuple[str, ...] = ("canny", "depth", "pose")
EDIT_SEEDS: tuple[int, ...] = (0, 1)
