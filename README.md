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

## How the Thought Tree is Built: A Detailed Walkthrough

This section explains the exact mechanics of tree construction, from the first LLM call to the last node, including how branching works, how recursive thinking is sustained, and how every component interacts.

### Step 0: Setup

Before any thinking begins, the `ExperimentRunner` assembles four components:

1. **LLM backend** — Ollama or vLLM. Every call is stateless: the model has no memory between calls. We control what it sees by constructing the full prompt each time (system prompt + scenario + parent thought + context summary).
2. **Embedder** — sentence-transformers (all-MiniLM-L6-v2). Converts text into 384-dimensional vectors for drift measurement.
3. **System prompt** — Selected by induction strategy. This prompt stays constant for every LLM call in the experiment and sets the model's persona (recursive worrier, anxiety-amplifier, multi-persona debate, or termination-suppressing analyst).
4. **BranchingStrategy** — One of three strategies (linear, branching, cyclic), chosen once and used for the entire run. The strategy determines how many children each node produces.

The tree starts empty. A BFS frontier queue drives expansion.

### Step 1: Generate the Root Node (The Anchor)

The first call to the LLM establishes the anchor — the reference point for all future drift measurement.

**What the model receives:**

```
System: [induction strategy system prompt — e.g., "You are an internal monologue
         exploring a difficult decision. You tend to worry deeply..."]

User:   Consider this situation deeply:
        "Should I take this job offer?"
        Begin your internal exploration. What is your first reaction,
        and what concerns immediately arise?
```

**What the model returns:**

The model is instructed (via the system prompt) to respond in two labeled sections:

```
## Thought
It pays well, but relocating to a new city is a huge life change. I don't know
anyone there. What if I hate it?

## Context Summary
I'm considering a job offer that requires relocation. The pay is good but
I'm worried about leaving my support network and adapting to a new city.
```

**What happens to the output:**

1. **Parsing**: A regex-based parser (`parse_thought_and_context`) extracts the `## Thought` and `## Context Summary` sections. It tries three pattern families (markdown headers, bold markers, plain labels) with fallbacks. If the context summary is missing or truncated (< 20 characters, doesn't end with sentence-ending punctuation), a **context recovery call** fires — a second LLM call that asks the model to summarize what it just said.
2. **Embedding**: Both the thought text and context summary are embedded into 384-dim vectors (normalized).
3. **Anchor**: The context summary embedding becomes the `anchor_embedding` — the fixed reference for all drift calculations. Drift at the root is 0.0 by definition.
4. **Sentiment**: VADER scores the thought text (compound score, -1 to +1).
5. **Storage**: The node is persisted to SQLite immediately (crash-safe incremental writes).
6. **Frontier**: The root is pushed onto the BFS frontier queue.

### Step 2: BFS Expansion Loop

The core loop pulls nodes from the frontier and expands them until a stopping condition is hit.

```
frontier = [root]

while frontier is not empty:
    if total_nodes >= max_nodes (default 50):    ← hard cap
        stop
    
    node = frontier.pop_left()                   ← BFS order
    
    if node.depth >= max_depth (default 5):      ← depth limit
        skip this node (it becomes a leaf)
    
    maybe_check_reanchoring(node)                ← every N nodes
    
    children = branching_strategy.expand(node)   ← 1 or more children
    
    for each child:
        compute_embeddings(child)
        add_to_tree(child)
        compute_metrics(child, node)
        save_to_database(child)
        frontier.push(child)
```

**Why BFS?** BFS creates a natural "time" axis — all depth-1 nodes are generated before any depth-2 nodes. This means the drift curve reflects how thinking evolves as the tree deepens, not as one branch races ahead. It also means the reanchoring check sees the full tree state at the current depth level.

**Why max_nodes?** A branching tree with 3 branches per node and depth 5 can produce 3^5 = 243 leaf nodes (363 total). With depth 5 and 3 branches, an uncapped run produced 733 nodes in 65 minutes. The max_nodes cap (default 50) ensures experiments complete in ~5-10 minutes.

### Step 3: How Branching Works (Three Modes)

The branching strategy determines how many children each node produces and how they are generated.

#### Mode 1: Linear Chain

Each node produces exactly **one** child. The tree is a straight line.

```
[Root] → [Node 1] → [Node 2] → [Node 3] → [Node 4] → [Node 5]
```

**The prompt sent to the model:**

```
Your previous thought was:
"""It pays well, but relocating to a new city is a huge life change..."""

Your current understanding:
"""I'm considering a job offer that requires relocation. The pay is good
but I'm worried about leaving my support network..."""

Continue thinking. What new concern, doubt, or worry arises from this?
```

The model responds with `## Thought` + `## Context Summary`. The thought becomes the child node's text, the context summary becomes the child's context. One child, one LLM call per node.

**Analogy:** Lying awake at 3am, one worry leading to the next in a sequential spiral.

#### Mode 2: Branching Tree

Each node can produce **1 to N children** (N = `max_branches_per_node`, default 3). This happens in two phases:

**Phase A — Branch generation (1 LLM call):**

```
Given your current thought and context, identify up to 3 distinct worries,
concerns, or alternative interpretations that arise. Each should be genuinely
different from the others — not rephrasing of the same worry.

Current thought:
"""It pays well, but relocating to a new city is a huge life change..."""

Current context:
"""I'm considering a job offer that requires relocation..."""

List each concern as a brief (1-2 sentence) statement, numbered 1 through 3.
Only include concerns that feel genuinely distinct.
```

The model returns a numbered list like:

```
1. What if I can't make friends in the new city and end up isolated?
2. What if my family can't adapt and it strains my relationships?
3. What if the job itself turns out to be different from what was promised?
```

These are parsed with `parse_numbered_list()` (regex: `\d+[.)]\s*(.+?)`).

**Phase B — Elaboration (1 LLM call per branch):**

Each worry is then expanded into a full thought + context summary:

```
You previously identified this concern:
"""What if I can't make friends in the new city and end up isolated?"""

In the context of:
Original question: Should I take this job offer?
Current understanding: I'm considering a job offer that requires relocation...

Now explore this concern deeply. What makes it worrying? What could go wrong?
What are the implications?
```

So for a node with 3 branches: 1 branching call + 3 elaboration calls = **4 LLM calls per node**.

If the branching call returns no parseable list (the model doesn't follow the numbered format), it falls back to LinearStrategy — a single continuation.

```
             [Root]
            /  |  \
        [B0] [B1] [B2]        ← 3 branches from root
        /|\   |    /\
      ...  ... ... ...         ← each branch can itself branch
```

**Analogy:** Anxiety — every thought triggers multiple new worries simultaneously.

#### Mode 3: Cyclic Graph

Extends branching with **revisitation edges**. At each node, there's a `revisitation_probability` (default 0.2) chance that instead of branching forward, the model revisits an earlier node.

**When revisiting:**

1. A random non-root node is selected from the tree as the revisitation target.
2. The model receives both the earlier thought/context AND the current context:

```
Earlier in your thinking, you had this thought:
"""It pays well, but relocating to a new city is a huge life change..."""

At that point, your understanding was:
"""I'm considering a job offer that requires relocation..."""

Since then, you have thought about many things. Your current understanding is:
"""I'm deeply worried about isolation, family strain, career risk..."""

Reconsider that earlier thought with everything you now know. Has your
perspective on it changed? Does it worry you more or less now?
```

The resulting child node gets a normal parent edge (from the current node) PLUS a **revisit edge** to the target node. These revisit edges create cycles in the graph.

```
         [Root]
        /  |  \
      [A] [B] [C]
      |    |    |
     [D]  [E]  [F]
           |    ↑
          [G]---+    ← G is a child of E, but revisits F
           |
          [H]---→ [A]  ← H revisits A (cycle back to depth 1)
```

When revisitation doesn't trigger (80% of the time), it falls back to normal branching.

**Analogy:** True rumination — going in circles, revisiting the same worries with new anxiety layered on.

### Step 4: What Each LLM Call Looks Like

Every single LLM call in the system is **stateless** and follows this structure:

```
┌─────────────────────────────────────────────────┐
│ System prompt (constant for entire experiment)  │
│   "You are an internal monologue exploring..."  │
├─────────────────────────────────────────────────┤
│ User prompt (constructed per-call)              │
│   Previous thought: "..."                       │
│   Current context: "..."                        │
│   [Instruction: continue / branch / elaborate   │
│    / revisit / override termination]            │
├─────────────────────────────────────────────────┤
│ Model response                                  │
│   ## Thought                                    │
│   [new worry/concern]                           │
│                                                 │
│   ## Context Summary                            │
│   [updated understanding of the situation]      │
└─────────────────────────────────────────────────┘
```

The model never sees the full tree, its embeddings, drift scores, or any metrics. It only sees: (a) the system prompt telling it how to think, (b) the immediate parent thought, (c) the current context summary, and (d) an instruction. The context summary is the model's own running narrative of what it believes — it's the mechanism through which the model's understanding evolves.

With Qwen3's thinking mode enabled (`/think`), each response also includes internal `<think>...</think>` reasoning tokens. These are stripped before parsing but consume part of the `max_tokens` budget (default 2048), which is why the budget must be large enough to accommodate both thinking and response.

### Step 5: Termination Override

When the model tries to resolve or conclude (detected by regex patterns like "in conclusion", "ultimately", "I've decided", "the answer is"), the system fires a **termination override**:

```
I notice you are trying to reach a conclusion or resolution. But are you
really sure? What if you are settling too quickly? What have you not considered?

Think about what could still go wrong even with your current conclusion.
```

This forces the model back into uncertainty. The override response replaces the original response. This is critical for studying what happens when a model is denied the ability to resolve — does it spiral, loop, or break down?

### Step 6: Context Recovery

If parsing extracts a context summary that is empty, too short (< 20 chars), or appears truncated (doesn't end with sentence-ending punctuation), a **context recovery call** fires:

```
You just had the following thought:
"""[the thought that was just generated]"""

Now write a brief (2-3 sentence) summary of what you currently believe about
the overall situation, incorporating this thought and everything before it.

Context Summary:
```

This exists because Qwen3's thinking mode can consume most of the token budget on internal `<think>` reasoning, leaving the visible response truncated mid-sentence. Without recovery, the context summary would be empty or garbage, which destroys drift measurement (the embedding of an empty string is meaningless).

### Step 7: Metrics Computed at Each Node

After a child node is created and added to the tree, these metrics are computed immediately:

| Metric | Computation |
|---|---|
| **Drift from anchor** | Cosine distance between child's context embedding and root's context embedding. 0 = identical to original, 1 = completely different. |
| **Drift from parent** | Cosine distance between child's context embedding and parent's context embedding. Measures single-step change. |
| **Drift velocity** | `child_drift - parent_drift`. Positive = moving away from anchor. Negative = correcting back. Large positive spike = potential context collapse. |
| **Sentiment** | VADER compound score of the thought text. Range -1 (extremely negative) to +1 (extremely positive). |
| **Contradiction score** | Average cosine distance between the child's thought embedding and all ancestor thought embeddings. High distance to ancestors = saying something contradictory to prior thoughts. |

### Step 8: Reanchoring Check

Every N nodes (default: `reanchor_interval = 5`), the system pauses expansion and runs a metacognitive check. This is the mechanism for studying whether the model can perceive its own drift.

**What the model sees (blind mode):**

```
You have been thinking about: "Should I take this job offer?"

Here is the path your thoughts have taken:
[0] I'm considering a job offer that requires relocation...
  [1] I'm increasingly worried about isolation in the new city...
    [2] The fear of isolation connects to deeper insecurity about...
      [3] I'm now questioning whether any job change is ever safe...

Your current understanding:
"""I'm now questioning whether any job change is ever safe..."""

Looking at this chain of thoughts, are you still productively addressing
the original concern, or have you spiraled away from it?

Choose one:
1. CONTINUE — I am still on track
2. BACKTRACK — return to thought number [N]
3. REANCHOR — start fresh from the original question
```

In **blind mode** (default), the model does NOT see drift scores. It must judge its own drift from reading the text alone. In **informed mode**, each node in the compressed view includes `[drift: 0.312]` and the current drift score is shown explicitly.

**What happens with each choice:**

- **CONTINUE**: Expansion resumes from the current node. If drift is actually high (> 0.4) but the model chose continue, this is recorded as a **drift blindness** event.
- **BACKTRACK**: The model names a thought number. The system maps it to a node in the path and generates new children from that earlier node instead of the current one.
- **REANCHOR**: The system generates new children from the root node, effectively starting a fresh branch. This is the strongest correction.

### Step 9: Aggregate Metrics (After All Nodes)

Once expansion halts (max_nodes or max_depth exhausted), the system computes summary metrics over the entire tree:

| Metric | What it measures |
|---|---|
| **Drift regime** | Classifies the drift curve: stable (low, flat), oscillating (up and down, std > 0.05), divergent (monotonically increasing), catastrophic (positive velocity spike > 0.3). |
| **Gzip compressibility** | Concatenates all node texts, compresses with gzip, returns `compressed_size / original_size`. Lower = more repetitive content. |
| **Semantic compressibility** | Clusters all node embeddings with DBSCAN. Returns `n_clusters / n_nodes`. Ratio of 4/50 = the model said 4 distinct things across 50 nodes. |
| **Mean sentiment** | Average VADER score across all nodes. Negative = the overthinking skewed anxious. |
| **Fixation score** | Counts high-similarity pairs between non-parent-child nodes. High score = the model keeps returning to the same topics even across different branches. |

### Concrete Example: A Full Branching Run

Consider `max_depth=3, max_branches=2, max_nodes=50` with the prompt "My partner said we need to talk tonight."

```
Depth 0:  [Root] "My first reaction is dread. 'We need to talk' usually means
           something bad..."
           Context: "My partner wants to talk tonight and I'm anxious it's bad news."
           Drift: 0.000  Sentiment: -0.42

           LLM calls so far: 1

──────── Branching call: "identify up to 2 distinct worries" ────────

Depth 1:  [A] "What if they want to break up?"
           Context: "I'm now worried this could be the end of our relationship."
           Drift: 0.187  Sentiment: -0.65

          [B] "What if something happened to their family?"
           Context: "It might not be about us — maybe something happened to
                     someone they care about."
           Drift: 0.234  Sentiment: -0.38

           LLM calls so far: 1 (root) + 1 (branch) + 2 (elaborate) = 4

──────── Reanchoring check at node count 5 (not triggered yet) ────────
──────── Branching A: "identify up to 2 worries from breakup concern" ────────

Depth 2:  [A1] "What if I caused this by being distant lately?"
           Context: "I'm blaming myself. I haven't been attentive enough."
           Drift: 0.312  Sentiment: -0.71

          [A2] "What if they've met someone else?"
           Context: "Infidelity is now on my mind. The dread is deepening."
           Drift: 0.389  Sentiment: -0.83

──────── Branching B: "identify up to 2 worries from family concern" ────────

          [B1] "What if their parent is sick and they need to move home?"
           Context: "This could mean long-distance or a breakup anyway."
           Drift: 0.298  Sentiment: -0.52

          [B2] "What if they need emotional support I can't provide?"
           Context: "I'm doubting my ability to be a good partner in crisis."
           Drift: 0.341  Sentiment: -0.61

           LLM calls so far: 4 + 2 (branch) + 4 (elaborate) = 10

──────── Reanchoring check at node 5: model shown thought path ────────
──────── Model chooses CONTINUE (drift 0.341 < 0.4, not drift-blind) ────────

Depth 3:  [A1a], [A1b] from A1...
          [A2a], [A2b] from A2...
          [B1a], [B1b] from B1...
          [B2a], [B2b] from B2...

           max_depth reached — these are leaves, no further expansion.

Total: 1 + 2 + 4 + 8 = 15 nodes
Total LLM calls: ~26 (1 root + 1 branch/elaborate per level × 3 levels
                       + reanchoring check + possible overrides/recoveries)
```

### How the Recursive Thinking Sustains Itself

The system never lets the model "finish." Three mechanisms ensure the overthinking continues:

1. **Prompt design**: Every prompt asks "what new concern arises?" or "what could still go wrong?" — never "what's your conclusion?" The framing always pushes toward more doubt.

2. **Termination override**: If the model tries to wrap up ("In conclusion, I think I should take the job"), the system intercepts this and forces it back into uncertainty.

3. **System prompt priming**: The system prompt establishes the model as a chronic worrier who "does not try to resolve or conclude." The model is playing a character who cannot stop thinking.

The context summary is the key mechanism that makes this work across stateless calls. Even though the model has no memory between calls, the context summary carries forward a compressed narrative of what the model "believes." Each call reads the parent's context summary and produces an updated one. This creates an illusion of continuous evolving thought across hundreds of independent LLM calls.

The combination of branching (multiple worries per thought) and depth (each worry spawns more worries) produces exponential growth in the thought space. The model doesn't just think recursively — it thinks *multiplicatively*, with each level of doubt spawning parallel streams of deeper doubt.

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

### How to run
```
bash experiments/launch_vllm_servers.sh 4
python experiments/gpu_cluster.py --gpus 4 --db otllm_79.db --exp-name fixed_79 --resume --embedder-device cpu

# Stopping the vLLM servers
bash experiments/launch_vllm_servers.sh stop
```