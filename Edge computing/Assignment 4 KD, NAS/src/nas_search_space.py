"""
Q2 - searchable CNN architecture space.

An architecture is described by an ArchConfig (plain, JSON-serialisable dataclass):

    resolution      input resolution the network is trained / deployed at     {24, 28, 32}
    stem_channels   width of the first 3x3 convolution                        {16, 24, 32}
    stages          3 or 4 stages; each stage has
        depth       number of blocks (convolutional depth)                     {1, 2, 3}
        channels    number of filters                                          per-stage choice list
        kernel      kernel size                                                {3, 5, 7}
        block       block type                                                 conv | dwsep | mbconv
                        conv   : standard KxK conv-BN-act
                        dwsep  : depthwise separable = KxK depthwise conv + 1x1 pointwise conv
                        mbconv : inverted residual = 1x1 expand, KxK depthwise, 1x1 project
        expand      expansion ratio (mbconv only)                              {2, 4, 6}
        activation  activation function                                        relu | relu6 | hswish | silu | leaky_relu
    head_channels   1x1 conv width before global pooling                      {128, 256, 384}
    dropout         classifier dropout                                         {0.0, 0.1, 0.2}

The first stage keeps the spatial size; every later stage halves it (first block has stride 2).
The space covers ~10^12 distinct networks, far too many to enumerate, hence the evolutionary search.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass, field

RESOLUTIONS = [24, 28, 32]
STEM_CHANNELS = [16, 24, 32]
NUM_STAGES = [3, 4]
DEPTHS = [1, 2, 3]
STAGE_CHANNELS = [[16, 24, 32], [32, 48, 64], [64, 96, 128], [128, 160, 192]]   # choices for stage i
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dwsep", "mbconv"]
EXPANDS = [2, 4, 6]
ACTIVATIONS = ["relu", "relu6", "hswish", "silu", "leaky_relu"]
HEAD_CHANNELS = [128, 256, 384]
DROPOUTS = [0.0, 0.1, 0.2]


@dataclass
class StageConfig:
    depth: int
    channels: int
    kernel: int
    block: str
    expand: int
    activation: str


@dataclass
class ArchConfig:
    resolution: int
    stem_channels: int
    stages: list = field(default_factory=list)
    head_channels: int = 256
    dropout: float = 0.1

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ArchConfig":
        d = copy.deepcopy(d)
        d["stages"] = [StageConfig(**s) for s in d["stages"]]
        return ArchConfig(**d)

    def key(self) -> str:
        """Canonical string used to detect duplicate architectures (expand is irrelevant unless mbconv)."""
        d = self.to_dict()
        for s in d["stages"]:
            if s["block"] != "mbconv":
                s["expand"] = 0
        return json.dumps(d, sort_keys=True)

    def summary(self) -> str:
        st = " | ".join(f"{s.block}{'x' + str(s.expand) if s.block == 'mbconv' else ''} k{s.kernel} c{s.channels} "
                        f"d{s.depth} {s.activation}" for s in self.stages)
        return f"res{self.resolution} stem{self.stem_channels} [{st}] head{self.head_channels} drop{self.dropout}"


# ---------------------------------------------------------------------- sampling / variation operators
def sample_stage(i: int, rng: random.Random) -> StageConfig:
    return StageConfig(depth=rng.choice(DEPTHS), channels=rng.choice(STAGE_CHANNELS[i]), kernel=rng.choice(KERNELS),
                       block=rng.choice(BLOCKS), expand=rng.choice(EXPANDS), activation=rng.choice(ACTIVATIONS))


def sample_config(rng: random.Random) -> ArchConfig:
    n = rng.choice(NUM_STAGES)
    return ArchConfig(resolution=rng.choice(RESOLUTIONS), stem_channels=rng.choice(STEM_CHANNELS),
                      stages=[sample_stage(i, rng) for i in range(n)], head_channels=rng.choice(HEAD_CHANNELS),
                      dropout=rng.choice(DROPOUTS))


def mutate(cfg: ArchConfig, rng: random.Random, p: float = 0.25) -> ArchConfig:
    """Each gene is re-sampled with probability p; at least one gene always changes."""
    child = copy.deepcopy(cfg)
    while child.key() == cfg.key():
        if rng.random() < p:
            child.resolution = rng.choice(RESOLUTIONS)
        if rng.random() < p:
            child.stem_channels = rng.choice(STEM_CHANNELS)
        if rng.random() < p / 2:                                     # add / remove a stage
            if len(child.stages) == 3:
                child.stages.append(sample_stage(3, rng))
            else:
                child.stages.pop()
        for i, s in enumerate(child.stages):
            if rng.random() < p:
                s.depth = rng.choice(DEPTHS)
            if rng.random() < p:
                s.channels = rng.choice(STAGE_CHANNELS[i])
            if rng.random() < p:
                s.kernel = rng.choice(KERNELS)
            if rng.random() < p:
                s.block = rng.choice(BLOCKS)
            if rng.random() < p:
                s.expand = rng.choice(EXPANDS)
            if rng.random() < p:
                s.activation = rng.choice(ACTIVATIONS)
        if rng.random() < p:
            child.head_channels = rng.choice(HEAD_CHANNELS)
        if rng.random() < p:
            child.dropout = rng.choice(DROPOUTS)
    return child


def crossover(a: ArchConfig, b: ArchConfig, rng: random.Random) -> ArchConfig:
    """Uniform crossover: every global gene and every stage is inherited from a random parent."""
    pick = lambda x, y: copy.deepcopy(x if rng.random() < 0.5 else y)   # noqa: E731
    n = rng.choice([len(a.stages), len(b.stages)])
    stages = []
    for i in range(n):
        cands = [p.stages[i] for p in (a, b) if i < len(p.stages)]
        stages.append(copy.deepcopy(rng.choice(cands)))
    return ArchConfig(resolution=pick(a.resolution, b.resolution), stem_channels=pick(a.stem_channels, b.stem_channels),
                      stages=stages, head_channels=pick(a.head_channels, b.head_channels),
                      dropout=pick(a.dropout, b.dropout))


if __name__ == "__main__":
    r = random.Random(0)
    for _ in range(3):
        c = sample_config(r)
        print(c.summary())
        print("  mutated:", mutate(c, r).summary())
