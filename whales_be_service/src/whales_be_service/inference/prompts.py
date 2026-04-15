"""CLIP zero-shot prompts for the anti-fraud gate.

Tuning these prompts changes the gate's behaviour without retraining anything.
The order matters: the first ``len(POSITIVE_PROMPTS)`` rows of the cached text
embedding tensor are the positives; the remainder are negatives.
"""

POSITIVE_PROMPTS: tuple[str, ...] = (
    "a photo of a whale in the ocean",
    "an aerial photo of a whale",
    "a photo of a humpback whale",
    "a photo of a blue whale",
    "a photo of a killer whale",
    "a photo of a dolphin swimming",
    "a photo of a beluga whale",
    "a photo of a marine mammal in the sea",
    "a photo of a whale fluke or dorsal fin",
    "a photograph of a cetacean",
)

NEGATIVE_PROMPTS: tuple[str, ...] = (
    "a photo of text on a blank page",
    "a screenshot of a document",
    "a photo of a person",
    "a photo of a building",
    "a photo of a car",
    "a photo of a cat",
    "a photo of a dog",
    "a photo of food",
    "a photo of a landscape without animals",
    "a blank white image",
    "a photo of a fish",
    "a photo of a boat on water",
    "a photo of a shark",
    "an abstract pattern",
)

ALL_PROMPTS: tuple[str, ...] = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
NUM_POSITIVE = len(POSITIVE_PROMPTS)
NUM_NEGATIVE = len(NEGATIVE_PROMPTS)
