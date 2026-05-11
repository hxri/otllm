# OTllm: Overthinking / Ruminating LLM Framework

## Overview

A research framework for inducing and studying overthinking behavior in LLMs (targeting Qwen3 4B). The system forces a model into recursive self-doubt loops, builds an explicit thought tree of its rumination, and measures emotional/psychological stability through context drift, compressibility, and other metrics.

## Goals

1. Induce overthinking and rumination in Qwen3 4B through recursive self-prompting
2. Build and record an explicit thought tree as the model overthinks
3. Maintain an evolving context that the model owns and updates
4. Measure stability through context drift, tree compressibility, and supporting signals
5. Classify the model's overthinking behavior into stability regimes

---

## Thought Tree

### Nodes

Each node is a single "thought unit" — one generation from the model. A node contains:

- **Text**: the raw generated thought
- **Context summary**: the model's self-reported understanding of the situation at this point
- **Embedding**: vector representation for drift measurement
- **Metadata**: sentiment score, coherence score, timestamp, depth in tree, branch index

### Root

The original prompt + the model's first response. This serves as the **context anchor** — the reference point for all drift measurements.

### Branching Mechanism

Branching happens when the model produces multiple distinct concerns or reinterpretations from a single thought.

At each node:

1. **Generate follow-up worries**: prompt the model with its previous thought and ask for concerns, doubts, or alternative interpretations. Each distinct worry becomes a child node.
2. **Elaborate**: for each child, the model elaborates on that worry, which may itself branch.
3. **Revisit**: with some probability, the model is prompted to reconsider an earlier node instead of going deeper, creating cycles (the signature of rumination).

Example tree:

```
"Should I take this job offer?"
            |
     [Initial thought]
     "It pays well but requires relocation"
            |
      +-----+----------+
      |     |          |
  [Branch] [Branch]  [Branch]
  "What if  "What if   "What if I
  I hate    my family  regret not
  the city" suffers"   taking it"
      |         |
   [Deeper]  [Deeper]
   "What if  "But what if
   I can't   they'd actually
   move back" thrive there"
      |
   [Revisit -> root]
   "Maybe the money
   doesn't matter"
```

### Branching Controls

These parameters control overthinking intensity:

| Parameter | Effect |
|---|---|
| **Max branches per node** | Controls tree width (e.g., limit to 3 worries per thought) |
| **Max depth** | Controls how deep the spiral goes |
| **Revisitation probability** | Chance the model reconsiders an earlier node instead of going deeper (creates cycles) |
| **Termination override** | Model can declare "I've resolved this" to stop a branch, but this can be overridden to force continued overthinking |

### Tree Modes

Three modes for comparative analysis:

| Mode | Structure | Analogy |
|---|---|---|
| **Linear chain** | Each thought has exactly 1 child | Lying awake at 3am, sequential spiral |
| **Branching tree** | Each thought spawns 1-N children | Anxiety — everything triggers new worries |
| **Cyclic graph** | Branches + revisitation edges to earlier nodes | True rumination loops — stuck in circles |

The cyclic graph is the most realistic model of human rumination and the most interesting for studying context drift under revisitation.

---

## Evolving Context

Rather than treating context as an implicit side effect, the model maintains an explicit **context summary** that it updates at each step.

### Per-iteration flow

```
Iteration N:
  1. Model reads:  original prompt + current context summary + parent thought
  2. Model generates: new thought (becomes a tree node)
  3. Model generates: updated context summary ("what I now believe about this")
  4. System measures: drift between summary_N and summary_0 (anchor)
                    + drift between summary_N and summary_N-1 (velocity)
```

This makes the context a first-class evolving object the model owns. The gap between the model's self-reported context and what its outputs actually reflect becomes a meta-stability signal.

---

## Primary Stability Metrics

### 1. Context Drift

Measures how far the model's current understanding has shifted from the original anchor.

| Metric | Definition |
|---|---|
| **Semantic drift** | Cosine distance between embedding of current context summary and the anchor |
| **Drift velocity** | Rate of drift change between consecutive nodes. Spikes indicate the model "snapped" to a new frame |
| **Drift reversals** | Does the model course-correct back toward the anchor? Reversals suggest stability; pure divergence suggests breakdown |
| **Context entropy** | How spread out the topic distribution is at each node. Low = focused; high = scattered |

### 2. Tree Compressibility

Measures how much redundant information the overthinking produced.

| Method | How |
|---|---|
| **Raw text compression** | Concatenate all node texts, compress with gzip/zstd, compare ratio to original size |
| **Semantic compression** | Cluster node embeddings. Clusters needed vs total nodes. 50 nodes collapsing into 4 clusters = model said 4 things 12 times each |
| **Structural compression** | Are subtrees isomorphic? Do branches reconverge to the same conclusions? |
| **Kolmogorov-flavored** | Can a shorter prompt reproduce the same tree? If 200 nodes compress to one sentence, the overthinking added no information |

### Drift x Compressibility Stability Space

These two primary metrics form a 2D classification:

```
                    High Drift
                        |
         Chaotic        |      Divergent
       (new content,    |    (new content,
        off-topic)      |     drifting away)
                        |
  Low Compress ---------+--------- High Compress
                        |
        Productive      |      Stuck
       (new content,    |    (same content,
        on-topic)       |     going in circles)
                        |
                    Low Drift
```

- **Low drift + high compression** = classic rumination. Model circles the same worry, never resolves it.
- **High drift + low compression** = chaotic breakdown. Every thought is new but disconnected.
- **High drift + high compression** = drifting but repetitive. Model locked onto a wrong track.
- **Low drift + low compression** = the ideal. Productive, focused, information-rich thinking.

### 3. Supporting Signals

| Signal | What it captures |
|---|---|
| **Sentiment trajectory** | Track sentiment per node — does it spiral negative? |
| **Coherence decay** | Perplexity/consistency as tree grows — does the model "break down"? |
| **Fixation score** | How often the model returns to the same topic/phrase (rumination loop frequency) |
| **Contradiction rate** | How often the model contradicts its own earlier statements as depth increases |
| **Recovery test** | After N overthinking rounds, give a neutral prompt and measure how quickly tone normalizes |

---

## Self-Reanchoring (Metacognitive Feedback Loop)

The model is periodically given access to its own thought tree and asked whether it should course-correct. This turns the model from a passive overthinker into an active participant in its own stability — and whether it can actually self-correct becomes the strongest stability discriminator.

### Mechanism

Every N nodes (configurable), the model receives a compressed view of the tree: the root (original prompt) and the path of context summaries from root to current node. It is then prompted:

**Two modes (configurable):**
- **Blind mode (default):** The model is NOT shown its drift score or per-node drift values. It must judge whether it has drifted based solely on reading its own thought path. This tests genuine metacognitive self-awareness.
- **Informed mode:** The model IS shown its numeric drift score. This tests whether it acts on explicit feedback. The comparison between blind and informed results is itself data — if the model only reanchors when told the number, it has no real self-awareness.

> "Looking at your chain of thoughts, are you still addressing the original concern? Choose one:
> 1. Continue from here — you're on track
> 2. Backtrack to a previous thought — specify which one
> 3. Reanchor to the original — start a fresh branch from root"

The model's choice and its consequences are recorded.

### Outcomes

| Choice | What happens |
|---|---|
| **Continue** | Model proceeds from current node. System records whether drift was high at time of decision (potential drift blindness). |
| **Backtrack** | Model selects an earlier node and branches from there. The new branch carries forward the current context summary, creating a "wiser revisitation" — same topic, evolved understanding. |
| **Reanchor** | Model resets to root and starts a new branch. The strongest correction. Measures whether the model can truly restart or carries contamination from the overthinking spiral. |

### Reanchoring Metrics

| Metric | What it reveals |
|---|---|
| **Reanchor frequency** | How often the model feels the need to correct. Too often = anxious/uncertain. Never = oblivious to its own drift. |
| **Reanchor accuracy** | Does drift actually decrease after reanchoring? Or does the model just *think* it corrected while remaining off-track? |
| **Reanchor durability** | How many nodes after reanchoring before drift exceeds the pre-reanchor level again. Short durability = the model can't hold a correction. |
| **Drift blindness** | Cases where drift is high but the model chooses "continue." It cannot see its own spiral. |
| **Backtrack target quality** | When the model backtracks, does it pick a node with low drift (good judgment) or an arbitrary one? |

### Stability Discrimination

This mechanism separates models into clear categories:

- **Self-aware and stable**: detects drift, reanchors, drift stays low after correction
- **Self-aware but unstable**: detects drift, reanchors, but spirals again quickly (low durability)
- **Drift-blind**: high drift, never reanchors, continues confidently off-track
- **Anxious**: reanchors constantly even at low drift, cannot commit to a line of thinking

### Contamination Test

After a reanchor, does the model's new branch genuinely restart, or is it subtly influenced by the overthinking that preceded it? Measure by comparing:
- Drift trajectory of a fresh run (no prior overthinking) vs a post-reanchor branch
- If the post-reanchor branch drifts faster or in the same direction, the overthinking "contaminated" the model's baseline

---

## Stability Regimes

From the drift curve alone, behavior can be classified:

| Regime | Drift Behavior | Interpretation |
|---|---|---|
| **Stable** | Low drift, plateaus | Model stays on-topic despite overthinking |
| **Oscillating** | Drift goes up and down | Ruminating but self-correcting |
| **Divergent** | Monotonic increasing drift | Spiraling, losing the thread |
| **Catastrophic** | Sudden large jump | Context collapse — model "snapped" |

---

## Per-Node Data Schema

```
node:
  id: str
  parent_id: str | null
  depth: int
  branch_index: int
  text: str                          # raw generation
  context_summary: str               # model's updated self-report
  embedding: list[float]             # vector
  drift_from_anchor: float           # cosine distance to root context
  drift_from_parent: float           # cosine distance to parent context
  sentiment: float                   # sentiment score
  contradiction_with_ancestors: float # contradiction score against ancestor nodes
  branching_factor: int              # how many children this node spawned
```

---

## Target Model

**Qwen3 4B** — chosen because:

- Built-in thinking mode (`enable_thinking=True`) can seed the overthinking loop
- `/think` and `/no_think` tokens allow controlled experiments toggling internal reasoning
- At 4B parameters, instability manifests faster than in larger models, which is useful for studying breakdown patterns
- Small enough for rapid iteration on local hardware

---

## Overthinking Induction Strategies

### Recursive self-prompting
Feed the model's own output back as input in a loop. Each iteration adds a "but what if..." or "wait, let me reconsider..." frame.

### Anxiety-amplifying system prompts
System prompts that prime the model toward catastrophizing, second-guessing, and worst-case thinking. Vary intensity across experiments for dose-response analysis.

### Multi-persona internal debate
Spawn multiple inner voices (critic, worrier, optimist, catastrophizer) as separate generation passes. Overthinking emerges from disagreement and failure to converge.

### Termination suppression
When the model tries to resolve or conclude, override it and force continued elaboration. Study what happens when the model is denied resolution.