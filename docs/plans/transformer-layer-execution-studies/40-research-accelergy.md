# Accelergy — architecture-level energy/area estimation (MIT, ICCAD 2019)

**Scope of this note.** Primary sources only: the ICCAD'19 paper text (extracted from
`http://accelergy.mit.edu/paper.pdf`), the source repo `Accelergy-Project/accelergy` at `master`,
the plug-in repos, the tutorial/exercise repo `Accelergy-Project/timeloop-accelergy-exercises`,
and `timeloop.csail.mit.edu/v4`. Every YAML block below is verbatim from a repo file, path given.
Inferences are marked **[INFERENCE]**.

**Headline for our purposes:** Accelergy is *not a mapper, not a simulator, and not a
performance model*. It is a per-action energy/area accumulator with a very good separation of
concerns. Its entire job is: given (a) a component tree, (b) a definition of what each component's
actions cost, and (c) how many times each action happened, produce energy. Someone else must
supply (c). It has no notion of time, order, contention, or bandwidth.

---

## 1. The separation of concerns (the part worth evaluating)

### 1.1 The decomposition, exactly

Paper §2 ("High-Level Framework") and the repo README agree on three inputs / three outputs.

**Inputs** (README, `github.com/Accelergy-Project/accelergy/blob/master/README.md`, "Input files"):

1. **Architecture description** — the component *tree*: names, classes, attribute values.
   Top key `architecture` / `architecture_description`, `version: 0.4`, then `nodes:` (v0.4) or
   `subtree:` (older).
2. **Compound component class description** — user-defined component *classes*: attributes,
   subcomponents, and action definitions. Top key `compound_components`, `version`, `classes:`.
3. **Action counts** — per-component, per-action counts for one workload run.

**Outputs** (README `-f/--output_files` flag): `flattened_arch`, `ERT`, `ERT_summary`, `ART`,
`ART_summary`, `energy_estimation`. Note what is *not* in that list: no latency, no cycles, no
schedule, no utilization.

**The pivotal design decision** is that the flow *splits at the ERT*. Paper §2:

> "Automatic design exploration tools, such as Timeloop [16], require fast energy consumption
> evaluations. To enable the integration of Accelergy with such tools, the generated ERTs for a
> hardware design are saved, so that they can be reused for action counts from different workloads.
> This avoids re-parsing the design descriptions and re-querying the component estimators."

So there are two engines inside one tool:

- **ERT/ART generator**: `(architecture + compound components) → ERT.yaml, ART.yaml`.
  Expensive (invokes CACTI, etc.). Depends only on *hardware*, not on workload.
- **Energy calculator**: `(ERT.yaml + action_counts.yaml) → energy_estimation.yaml`.
  Cheap. Depends on *workload*, not on hardware internals.

README confirms all three modes are supported independently:
"Providing the **generated ERTs** and the **action counts** allows Accelergy to directly generate
energy estimations."

### 1.2 The real YAML

**ERT** — verbatim from
`timeloop-accelergy-exercises/workspace/example_designs/example_designs/simple_weight_stationary/ref_outputs/default_problem/timeloop-mapper.ERT.yaml`
(energies in pJ):

```yaml
ERT:
    version: '0.4'
    tables:
      - name: system_top_level.shared_glb[1..1]
        actions:
          - name: write
            arguments: {}
            energy: 26.156875
          - name: read
            arguments: {}
            energy: 32.377775
          - name: leak
            arguments: {}
            energy: 0.005353
          - name: update
            arguments: {}
            energy: 26.156875
      - name: system_top_level.mac[1..256]
        actions:
          - name: compute
            arguments: {}
            energy: 0.315
          - name: leak
            arguments: {}
            energy: 0.0036
      - name: system_top_level.DRAM[1..1]
        actions:
          - name: leak
            arguments:
                global_cycle_seconds: 1e-09
                action_latency_cycles: 1
            energy: 0.0
          - name: read
            arguments:
                global_cycle_seconds: 1e-09
                action_latency_cycles: 1
            energy: 512.0
```

That is the whole ERT format: a flat list of `(component instance pattern, action name,
action arguments) → scalar energy`. `[1..256]` is the instance-count notation; the energy is
**per instance per action**.

**ART** — same directory, `timeloop-mapper.ART.yaml`, verbatim:

```yaml
ART:
    version: '0.4'
    tables:
      - name: system_top_level.DRAM[1..1]
        area: 0.0
      - name: system_top_level.shared_glb[1..1]
        area: 429745.625
      - name: system_top_level.pe_spad[1..256]
        area: 1419.815
      - name: system_top_level.mac[1..256]
        area: 417.0
```

Even flatter: one scalar (um²) per component instance. No actions.

**Action counts** — verbatim from
`accelergy/test/tests/basic/data/action_counts.yaml`:

```yaml
action_counts:
  version: 0.2
  subtree:
    - name: design
      local:
        - name: scratchpad[0]
          action_counts:
            - name: leak
              counts: 100
            - name: fill
              arguments:
                address_delta: 0
                data_delta: 0
              counts: 1150
            - name: fill
              arguments:
                address_delta: 1
                data_delta: 1
              counts: 24
  local:
    - name: design.mac
      action_counts:
        - name: mac_random
          counts: 50
        - name: mac_gated
          counts: 100
```

Note `address_delta` / `data_delta` as action *arguments*: the same `fill` action is billed at
different rates depending on whether the address/data changed. This is the paper's "data property"
axis (§4) made concrete — the counter has to bin its counts by argument value, not just by action.

### 1.3 The interface, and who does what

The consumer side is three lines of Python. From
`accelergy/accelergy/energy_calculator.py` (verbatim):

```python
for component_name, action_counts_obj_list in self.action_counts.get_action_counts().items():
    component_energy = 0
    for action_count_obj in action_counts_obj_list:
        energy_per_action = ERT_entry_obj.get_action_energy(action_count_obj)
        component_energy = component_energy + energy_per_action * action_count_obj.get_action_count()
```

That is the entire cost model: `Σ counts × energy_per_action`. Nothing else.

**Who produces action counts?** Not Accelergy. Paper §2:

> "Accelergy takes in an architecture description and runtime action counts, which are based on a
> specific workload that is generated by performance models (e.g., cycle-accurate simulators or
> analytical models)."

Paper §5.2 confirms the authors wrote their own: "we built a parameterizable cycle-level DNN
simulator in Python as the performance model … users can use any simulator to generate the action
counts, as long as the generated statistics adhere to Accelergy's action counts format."

**In Timeloop+Accelergy specifically**, the split is:

- Accelergy generates `ERT.yaml` + `ART.yaml` from the architecture
  (`timeloop.csail.mit.edu/v4/parsing_and_intermediate_files/energy-and-area-reference-tables`:
  "intermediate files generated by Accelergy … the energy of each action by each component in the
  architecture and the area of each component").
- **Timeloop** derives the access counts analytically from the mapping and does the multiplication
  itself. Timeloop's stats page (`timeloop.csail.mit.edu/v4/output-formats/stats`) reports per-level
  "Scalar reads / fills / updates", "Energy (per-scalar-access)", and computes e.g.
  "the energy per compute and the number of computes (`0.56*49152`)".
- The MICRO'52 tutorial slide deck (`accelergy.mit.edu/micro52/02_accelergy.pdf`, slides 15–16)
  draws exactly this: Timeloop's `Mapper → Model` box emits *Energy / Performance / Area*, and
  Accelergy feeds it the ERT table `(Component name, Action name, Energy/action)`.

**[INFERENCE]** In the Timeloop pairing, the standalone `action_counts.yaml` path is essentially
unused — Timeloop consumes the ERT directly rather than round-tripping counts through Accelergy's
energy calculator. The `action_counts.yaml` schema is the interface for *everyone else* (custom
simulators, gem5 via `gem5-accelergy-connector`, or, hypothetically, us).

### 1.4 The plug-in interface

`accelergy/accelergy/plug_in_interface/interface.py` — five abstract methods, verbatim signatures:

```python
class AccelergyPlugIn(ListLoggable, ABC):
    @abstractmethod
    def primitive_action_supported(self, query: AccelergyQuery) -> AccuracyEstimation: ...
    @abstractmethod
    def estimate_energy(self, query: AccelergyQuery) -> Estimation: ...
    @abstractmethod
    def primitive_area_supported(self, query: AccelergyQuery) -> AccuracyEstimation: ...
    @abstractmethod
    def estimate_area(self, query: AccelergyQuery) -> Estimation: ...
    @abstractmethod
    def get_name(self) -> str: ...
```

A query is `AccelergyQuery(class_name, class_attrs, action_name, action_args)` — i.e. "how much
does an `SRAM(width=64, depth=16384, technology=45nm)` cost for a `read()`?"

**Arbitration** (`accelergy/plug_in_interface/query_plug_ins.py`, `get_best_estimate`):

```python
accuracies = sorted(accuracies, key=lambda x: x[1].value, reverse=True)
for plug_in, accuracy in accuracies:
    if not accuracy.success or accuracy.value == 0:
        continue
```

Each plug-in **self-reports** a 0–100 accuracy for each query; the highest self-reported accuracy
that successfully estimates wins. There is a `minimum_accuracy` / `min_accuracy` attribute users can
set to reject low-confidence plug-ins. This is a nice mechanism and a serious caveat — see §3.3.

---

## 2. Compound components — could a memtile or a shim BD be expressed?

### 2.1 The abstraction

Paper §3.3: "A compound component is defined as a high-level function unit that consists of several
primitive components or other high-level function units". To define a class you give
"(1) a set of attributes, (2) a set of sub-components, and (3) a set of compound action names and
definitions."

The composition rule (§3.3.3): "A compound action is defined as an aggregate of the lower level
sub-components' action types." Energy composes as a weighted sum over subcomponent actions, with
`repeat` (v0.3) / `energy_scale` / `action_share` (v0.4) as the multiplier and `area_share` /
`area_scale` for area. Attributes flow downward: a subcomponent attribute can be a literal, an
inherited parent attribute, or an arithmetic expression over parent attributes.

The schema, verbatim from
`timeloop-accelergy-exercises/workspace/cheatsheets/4_compound_component.yaml`:

```yaml
compound_components:   # REQUIRED top-level key
  version: 0.4         # REQUIRED version number
  classes:             # Compound component classes go below
  - name: component_name_here
    attributes: # Attributes listed here can be thought of as the "inputs" to the component
      attribute_name_1: 123 # Default values can be specified here
      attribute_name_2: "must_specify" # "must_specify" means that the user must specify a value
    subcomponents:
    - name: subcomponent_name_1
      class: class_of_subcomponent_1 # class must be defined by a plug-in or compound component
      area_scale: 1 # Area share scales the area of the subcomponent
      attributes:
        subcomponent_attribute_1: 123
    - name: subcomponent_name_2[1..123]   # Multiple subcomponents via a range
      class: class_of_subcomponent_2
    # Each component MUST have read, write, update, and leak actions
    actions:
    - name: read # Read action processes a value
      subcomponents:
      - name: subcomponent_name_1
        actions:
        - name: read
          energy_scale: 1 # energy_scale specifies how many times the action is performed
          arguments: {arg1: 123, arg2: 456}
      - name: subcomponent_name_2[0]
        actions:
        - name: read
          energy_scale: 1
    # Specify empty subcomponents to have zero-energy actions
    - {name: write, subcomponents: []}
    - {name: update, subcomponents: []}
    - {name: leak, subcomponents: []}
```

A real one —
`timeloop-accelergy-exercises/.../_components/smartbuffer_SRAM.yaml`, verbatim:

```yaml
compound_components:
  version: 0.4
  classes:
  - name: smartbuffer_SRAM
    attributes:
      technology: "must_specify"
      width: "must_specify"
      depth: "must_specify"
      n_rw_ports: 1
      global_cycle_seconds: "must_specify"
    subcomponents:
    - name: storage
      class: SRAM
      attributes:
        width: width
        depth: depth
        n_rw_ports: n_rw_ports
        technology: technology
        global_cycle_seconds: global_cycle_seconds
    - name: address_generators[0..1]
      class: intadder
      attributes:
        n_bits: max(1, ceil(log2(depth))) if depth >= 1 else 1
    actions:
    - &write_action
      name: write
      subcomponents:
      - name: storage
        actions: [{name: write}]
      - name: address_generators[0]
        actions: [{name: add}]
    - name: read
      subcomponents:
      - name: storage
        actions: [{name: read}]
      - name: address_generators[1]
        actions: [{name: add}]
    - name: leak
      subcomponents:
      - name: storage
        actions: [{name: leak}]
      - name: address_generators[0..1]
        actions: [{name: leak}]
    - name: update
      << : *write_action # Update is the same as write
```

Note the arithmetic in attributes (`max(1, ceil(log2(depth)))`) and the YAML-merge trick for
"update behaves like write". These are the real ergonomics.

### 2.2 A memtile with N DMA channels — what it would take

Yes, it is expressible, and mechanically it is easy. **[INFERENCE — I wrote this; it is not from
the repo, and I have not run it]**:

```yaml
compound_components:
  version: 0.4
  classes:
  - name: aie2p_memtile
    attributes:
      technology: "must_specify"
      global_cycle_seconds: "must_specify"
      depth: 32768          # 512 KB / 16B lines, per memtile
      width: 128            # bits per access
      n_dma_channels: 12    # per-tile DMA channel count
      n_bds: 48             # buffer descriptors
    subcomponents:
    - name: storage
      class: SRAM
      attributes: {width: width, depth: depth, n_rw_ports: 2,
                   technology: technology, global_cycle_seconds: global_cycle_seconds}
    - name: dma_addr_gen[0..n_dma_channels-1]
      class: intadder
      attributes: {n_bits: max(1, ceil(log2(depth)))}
    - name: bd_table
      class: regfile
      attributes: {width: 128, depth: n_bds}
    actions:
    # A "strided BD push" of one 4-D descriptor moving `n_words` words
    - name: dma_transfer
      arguments:
        n_words: 1..65536
        n_dims: 1..4                 # how many stride levels are actually walked
      subcomponents:
      - name: storage
        actions: [{name: read, energy_scale: n_words}]
      - name: dma_addr_gen[0]
        actions: [{name: add, energy_scale: n_words * n_dims}]
      - name: bd_table
        actions: [{name: read, energy_scale: 1}]
    - name: leak
      subcomponents:
      - name: storage
        actions: [{name: leak}]
      - name: dma_addr_gen[0..n_dma_channels-1]
        actions: [{name: leak}]
      - name: bd_table
        actions: [{name: leak}]
    - {name: read,  subcomponents: [{name: storage, actions: [{name: read}]}]}
    - {name: write, subcomponents: [{name: storage, actions: [{name: write}]}]}
    - {name: update, subcomponents: [{name: storage, actions: [{name: update}]}]}
```

A shim tile issuing a strided BD is the same shape: a `DRAM`-class subcomponent for the
off-chip access plus an address-generation subcomponent whose action count scales with the number
of stride dimensions walked. The `arguments:` mechanism (paper §3.3.3, "actions with arguments") is
exactly the hook for "energy depends on the descriptor's dimensionality/stride pattern".

**What it would actually take, honestly:**

1. **The structural description is the cheap part** (an afternoon). It is genuinely expressive
   enough for AIE2P's memtile/shim/core hierarchy.
2. **The primitive energies are the expensive part and we do not have them.** Every leaf must
   resolve to a `class` some plug-in can price. For a 4 nm-class AIE2P, no shipped plug-in has
   credible data (§3.4).
3. **The action counts are the part we'd have to build.** Nothing in Accelergy tells you how many
   times a memtile DMA fired for a given tiling. That's the whole mapping problem, and it is
   out of scope for Accelergy by construction.
4. **`energy_scale: n_words` linearity is a modeling assumption we would be inventing**, and we
   would have no way to validate it without per-component power measurement we don't have.

**[INFERENCE]** Items 2 and 3 dominate. The compound-component YAML is ~5% of the work.

---

## 3. Where the numbers come from — be skeptical here

### 3.1 The plug-ins actually shipped

| Plug-in | Repo | Basis | Technology |
|---|---|---|---|
| CACTI | `accelergy-cacti-plug-in` (→ `hwcomponents-cacti`) | CACTI 7 analytical memory model | **22–180 nm**, hard-asserted in code |
| Aladdin | `accelergy-aladdin-plug-in` | Aladdin [ISCA'14] datapath energy tables | **40 nm only** ("The 40nm energy data is stored in the `data` folder") |
| Table-based | `accelergy-table-based-plug-ins` (deprecated) | user CSV tables | whatever you supply |
| Library | `accelergy-library-plug-in` | "a library of components from published works"; "citations are required for all entries" | mixed, per-entry |
| NeuroSim / ADC / NVMExplorer / McPAT | separate repos | domain-specific | varies |
| Dummy | shipped in-tree | returns `1 pJ` for everything | none — test only |

The dummy plug-in is worth naming because it is a footgun:
`share/estimation_plug_ins/dummy_tables/dummy_table.py` returns `Estimation(1, "p")` for every
non-leak action and `Estimation(1, "u^2")` for every area, gated on `technology == -1`. Timeloop
even ships a processor called `enable_dummy_table.py`. **[INFERENCE]** Published numbers produced
with the dummy table enabled would be meaningless, and nothing in the output file marks them —
you'd have to read `ERT_summary.yaml` to see which plug-in answered.

### 3.2 Technology extrapolation is ad hoc, and this bites us directly

From `accelergy-cacti-plug-in/cacti_wrapper.py`, verbatim:

```python
def _interp_technology(self):
    supported_technologies = [22, 32, 45, 65, 90]
    # Interpolate. Below 16, interpolate energy with square root scaling (IDRS 2022),
    # area with linear scaling.
    if self.technology < min(supported_technologies):
        scale = self.technology / min(supported_technologies)
        ...
        read_energy *= scale**0.5
        write_energy *= scale**0.5
        update_energy *= scale**0.5
        area *= scale
        # finfets have approx. 21% less leakage power
        leak_power *= scale**0.5 * 0.79
```

For a 4 nm-class node this is `scale = 4/22 ≈ 0.18`, so read energy is CACTI's 22 nm number times
`√0.18 ≈ 0.43`, and area times `0.18`. That is a two-line closed-form extrapolation **5.5× outside
the model's supported range**, justified in the comments by a wikichip news article and a 2009
FinFET paper. It is a defensible engineering guess. It is not an artifact.

Similar hand-rolled scaling exists for geometry (`cacti_wrapper.py`, `_interp_size`):

```python
# width: Area,dynamic,leakage energy scale linearly. Delay does not scale.
# depth: Area,leakage energy scale linearly. Delay, dynamic energy scale with 1.56/2
read_energy *= (widthscale * 0.7 + 0.3) * (depthscale ** (1.56 / 2))
```

### 3.3 "Accuracy" is a self-declared constant

The plug-in arbitration in §1.4 sorts by an accuracy number the plug-in *asserts about itself*.
The CACTI wrapper's first line of source is literally:

```python
# in your metric, please set the accuracy you think CACTI's estimations are
```

and the class declares `percent_accuracy_0_to_100 = 80`. The Aladdin plug-in README documents
`ALADDIN_ACCURACY`, "a configurable parameter in `aladdin_table.py` with a default value of 70".

**These are opinions typed into source files, not measured calibration.** The mechanism is sound —
it lets you express "prefer the silicon-calibrated table over the analytical model" — but the
numbers carry no evidentiary weight.

### 3.4 The paper's accuracy claim, read carefully

The abstract claims "Accelergy achieves 95% accuracy on Eyeriss". What the paper actually did (§5):

- **Ground truth is post-layout simulation, not silicon.** §5.1: "The accelerator design is written
  in RTL, synthesized, and placed-and-routed in a 65nm technology." §5.3: "The total estimated
  energy is within 5% of the post-layout results." Relative per-block breakdown: "within 8%".
  Fig. 7 (PE array total): ground truth 100%, Accelergy 95%, Aladdin 88%, fixed-cost 78%.
  **No chip measurement is reported anywhere in the paper.**
- **The primitive energies were calibrated from the same flow as the ground truth.** §5.1: "The
  energy-per-action values in the library are generated using post-layout simulations of small
  modules (e.g., the MAC component)." So the estimator's leaf costs and the reference number come
  from the same 65 nm PnR flow on the same design's submodules.
- **The action counts came from a simulator the authors wrote for this experiment.** §5.2: "we
  built a parameterizable cycle-level DNN simulator in Python as the performance model."
- **N = 1 design, 1 workload.** AlexNet weights + ImageNet inputs, quantized to 16 bits (§5.3).

**[INFERENCE, but well-supported]** The 95% figure validates *the compound-component + fine-grained
action-type methodology*, given a primitive table calibrated against the same technology and flow.
It does **not** validate "install Accelergy with the CACTI and Aladdin plug-ins, describe a new
accelerator, get 95%". Fig. 7 arguably shows the opposite: Aladdin-as-methodology lands at 88% on
the design it was applied to. And because counts and costs both came from the authors' own models,
a count error and a cost error could partially cancel; the experiment cannot separate them.

**Verdict on artifact quality:** the ICCAD'19 number *does* have an artifact behind it (a real
65 nm PnR'd Eyeriss), which is better than most architecture-level tools. But it is one point, in
one technology, self-consistently calibrated, against simulation rather than a power rail.

---

## 4. What Accelergy does NOT model

**It does not model time. At all.** Evidence, in descending order of dispositiveness:

1. **The output list has no time in it.** README `-f`: `flattened_arch, ERT, ERT_summary, ART,
   ART_summary, energy_estimation`.
2. **The cost model is a dot product** (`energy_calculator.py`, §1.3):
   `Σ count × energy_per_action`. There is no ordering, no queue, no dependency, no clock.
3. **The paper outsources time explicitly.** §2: action counts are "generated by performance models
   (e.g., cycle-accurate simulators or analytical models)."
4. **The word "latency" appears in the paper only in a citation title** (McPAT). "Contention",
   "stall", "bandwidth", "throughput" (as a modeled quantity) do not appear at all.

Corollaries:

- **No contention, no stalls, no arbitration.** If two DMA channels collide, Accelergy will never
  know; the counts it receives are already whatever they are.
- **No bandwidth.** Bandwidth attributes exist in the *Timeloop v4 architecture* schema
  (`read_bandwidth`, `write_bandwidth` in `cheatsheets/3_architecture.yaml`), but those are
  Timeloop's constraint knobs, not Accelergy's. Accelergy just forwards unknown attributes to
  plug-ins.
- **Area was not in the paper.** The ICCAD'19 text never mentions ART or area estimation; the repo
  README's changelog says "Update v0.3 — Addition of area reference table generation". **This is a
  clean example of tool ≠ paper: cite the repo for ART, not the paper.**
- **Time enters only as a passed-through parameter.** `global_cycle_seconds` and
  `action_latency_cycles` appear as ERT action *arguments* (see the DRAM entry in §1.2). Per
  `timeloop.csail.mit.edu/v4/input-formats/variables`, `global_cycle_seconds` is "the global cycle
  time in seconds… used to specify the cycle time of the entire system" and `action_latency_cycles`
  is "the latency of an action in cycles", default 1. **[INFERENCE]** Their purpose is to let
  plug-ins convert leakage *power* into per-cycle leakage *energy*, and to let a long action
  accumulate proportional leakage. They are inputs Accelergy is told, not quantities it derives.
- **The 2025-era successor moves slightly.** `Accelergy-Project/hwcomponents` ("HWComponents
  provides area, energy, latency, and leak power estimates for hardware components") does add
  per-component latency models. **[INFERENCE]** That is still a per-component number, not a system
  schedule; it does not turn the stack into a performance model.

**Plainly stated, as requested: Accelergy is a per-action energy accumulator that requires someone
else to supply the counts.** Everything interesting about *when* things happen is upstream of it.

---

## 5. Is an analytical energy model useful to a project that can already measure?

Not assuming yes. Both sides, honestly.

### 5.1 Where it is strictly worse than what we already do

- **Total energy for a real layer.** If we can read a power rail or an on-die energy counter during
  a real run, that number is ground truth and Accelergy's is a model. Adopting Accelergy here trades
  a measurement for an estimate. Strictly worse.
- **Anything involving contention or the real DMA schedule.** Our wall-clock latency already
  contains every stall, every BD conflict, every shim arbitration event. Accelergy contains none of
  them and cannot be made to.
- **DRAM traffic.** We count bytes. Accelergy would multiply our count by an
  `LPDDR4: 8 pJ/bit`-style constant from `cacti_wrapper.py`'s `type2energy` dict. That's our own
  number times a literature constant — it adds a unit conversion, not information.
- **Calibration cost.** To make any Accelergy number trustworthy on AIE2P we would need per-component
  energies at a 4 nm-class node. Those do not exist publicly, the CACTI path extrapolates 5.5×
  out of range (§3.2), and the Aladdin path is 40 nm. **[INFERENCE]** The realistic path is to
  *fit* the ERT to our own measurements — at which point the "model" is a regression on our data
  and its predictions outside the fitted region are unwarranted.
- **Our stated gap is a balance instrument and a mapping cost model.** Accelergy is neither. It
  would not tell us whether a tiling is compute-bound or DMA-bound.

### 5.2 Where it adds something we genuinely cannot measure

- **Attribution.** This is the real one. A wall-clock latency and a total-energy reading are
  scalars; they cannot tell us "63% of this layer's energy is memtile→core L1 traffic and 11% is the
  MACs". Accelergy's Eyeriss result is precisely this capability (§5.4, Figs. 8–9: per-PE and
  intra-PE breakdown), and its selling point over the "fixed-cost" baseline is that it captured
  breakdowns the coarse method got wrong. We have no instrument that attributes cost to a *component*
  rather than to a *time interval*.
- **Counterfactuals before building.** "What if the memtile were 2× deeper", "what if we replicated
  4× across columns" — these are questions about designs we cannot build. An ERT + a count model
  answers them; hardware cannot.
- **Forcing the count model to exist.** **[INFERENCE — the strongest argument in my view]** To feed
  Accelergy we would have to write a thing that predicts, for a given tiling, how many memtile
  reads / shim BD issues / core L1 accesses occur. *That* artifact is a balance instrument, and we
  currently lack it. The energy multiplication at the end is almost incidental.
- **Cross-checking measurements.** An independent analytic prediction that disagrees with a
  measurement is a bug detector. Given our MEMORY note on "claims without artifacts" and the pmode
  incident, a second, differently-derived number has real value.

### 5.3 My honest read

**Do not adopt Accelergy.** The tool's value is concentrated in (a) primitive energy libraries we
cannot use at our node and (b) an architecture for separating hardware cost from workload counts.
We would get (b) for free by copying the pattern and none of (a).

But **build the action-count layer**, because that is the missing balance instrument and it happens
to be exactly Accelergy's input format. If we ever want energy attribution later, an ERT is 30 lines
of YAML on top of it.

---

## 6. What transfers as a design pattern

Ranked by value to us.

1. **The ERT indirection — hardware cost separated from workload counts, with a persisted table
   between them. [HIGH]** The reason this is good is not energy; it is that it makes the expensive
   part cacheable and the cheap part sweepable. Our analogue: a per-component table of
   `(component, action, arguments) → measured cost` — e.g. `memtile.dma_transfer(n_words, n_dims)
   → ns`, `shim.bd_issue(strided) → ns`, populated from microbenchmarks on real silicon. Then a
   mapping sweep is a dot product against that table instead of 500 hardware runs. **This is the
   single most transferable idea, and it is the one we can populate with measurements rather than
   models.**

2. **Actions with arguments — bin the counter by the thing that changes the cost. [HIGH]** Paper
   §3.3.3 and the `address_delta`/`data_delta` example. The insight generalizes past energy: a
   `dma_transfer` is not one cost, it is a cost function of `(n_words, n_dims, stride)`, and a
   counter that reports a single scalar "number of DMA transfers" has already destroyed the
   information you need. Our BD-stride walls make this concrete — the same "one transfer" costs
   wildly different amounts depending on descriptor shape. Design our counters to emit
   `(action, args) → count` from the start.

3. **The action-count schema as a stable interface between "what ran" and "what it cost". [MEDIUM-HIGH]**
   Adopting a tree-shaped `component → action → args → count` format means the producer (a compiler
   pass reading the AIR/AIE IR, or a runtime trace) and the consumer (a cost model, a balance
   report, a spreadsheet) can evolve independently. Cheap to adopt; we can literally reuse the YAML
   shape in §1.2.

4. **Compound components with attribute inheritance and arithmetic. [MEDIUM]** Worth stealing the
   *idea* that a memtile is a named composite whose cost is defined once and instantiated 8× with
   different attributes, and that its cost decomposes into subcomponent costs. Worth being skeptical
   of the *implementation*: string-typed arithmetic in YAML
   (`max(1, ceil(log2(depth))) if depth >= 1 else 1`) plus Jinja templating plus YAML anchors is a
   configuration language reinvented badly. If we build this, build it in Python.

5. **Plug-in arbitration by declared accuracy, with a `minimum_accuracy` floor. [MEDIUM — but
   invert it]** The mechanism ("multiple estimators can answer; pick the most trustworthy") is
   genuinely good and directly applicable to us: for a given `(component, action)` we might have a
   measured microbenchmark, a fitted regression, or an analytic guess, and we want the measured one
   to win automatically and the provenance to be recorded. **But do not copy the self-declared
   constant.** Rank by *evidence class* (measured on this silicon > fitted from our measurements >
   literature) and make the output record which class answered — Accelergy's `ERT_summary.yaml`
   does record the responding plug-in, which is the right instinct.

6. **Explicit `leak` / idle actions. [LOW-MEDIUM]** The v0.4 schema *requires* every component to
   define `read`, `write`, `update`, and `leak`. **[INFERENCE]** The forcing function is the useful
   part: it makes "this component was idle for N cycles" a first-class, countable fact rather than
   something that silently disappears. Our analogue is idle/stall time per tile, which is exactly
   the balance signal we lack.

**What not to steal:** the technology-scaling extrapolations (§3.2), the shipped primitive energy
tables (wrong node), and the dummy-table default that can silently produce meaningless output.

---

## Comparable summary

- **Data-space representation** — *Not applicable; it is not a mapper.* Accelergy has no concept of
  tensors, dataspaces, or iteration spaces. Timeloop owns those; Accelergy sees only a component
  tree and integer action counts. (Sparsity reaches it only indirectly, as differently-named action
  types like `zero_gated_MAC` — paper §4.)

- **Mapping-space representation** — *Not applicable.* No loop nests, no tiling, no permutation.
  The `constraints:` blocks in `cheatsheets/3_architecture.yaml` (factors, permutation, bypass) are
  **Timeloop's** schema, not Accelergy's; Accelergy consumes only the `attributes:` of the same file.

- **Legality model** — *Not applicable in the mapping sense.* The only validity checks are
  structural: every referenced subcomponent class must resolve to a plug-in or another compound
  class; `"must_specify"` attributes must be supplied; every component must define `read`/`write`/
  `update`/`leak`; and at least one plug-in must return non-zero accuracy for each
  `(class, action)` query or estimation fails.

- **Search strategy** — *Not applicable; it does not search.* It is a pure evaluator invoked by a
  searcher (Timeloop's mapper). The ERT-caching design (paper §2) exists specifically to make it
  cheap enough to sit inside someone else's search loop.

- **Cost model** — **Energy and area only, as a linear per-action accumulator:**
  `E = Σ_components Σ_actions count(action, args) × ERT[component][action][args]`
  (`energy_calculator.py`). Leaf costs come from pluggable estimators (CACTI 22–180 nm, Aladdin
  40 nm, user tables). Zero time, zero contention, zero bandwidth modeling.

- **Multi-op support** — *Not applicable at the tool level; trivially true at the arithmetic level.*
  Accelergy has no notion of operators or layers, so counts from any number of ops can simply be
  summed into one action-count file. It contributes nothing to reasoning about fusion, inter-op
  pipelining, or cross-op reuse — those change the *counts*, which is upstream.

- **Single most transferable idea** — **The ERT indirection: persist a
  `(component, action, arguments) → cost` table derived from hardware, and make everything
  downstream a dot product against it.** For us the table holds measured nanoseconds and bytes
  rather than modeled picojoules, which makes it *stronger* than Accelergy's, not weaker. Runner-up
  and inseparable from it: actions carry **arguments**, so a strided-BD cost is a function of the
  descriptor shape rather than one averaged scalar.

- **Single biggest mismatch with our target** — **It models the wrong quantity.** We need a
  time/balance instrument for a 32-core spatial array with real DMA contention; Accelergy models
  energy and explicitly delegates all timing to a performance model we would still have to write.
  Compounding that: its primitive energy libraries stop at 22 nm (CACTI) and 40 nm (Aladdin) and
  reach our node only through a `scale**0.5` extrapolation ~5.5× out of range, so even its one
  quantity would be uncalibrated for AIE2P.

---

## Sources

Paper
- Y. N. Wu, J. S. Emer, V. Sze, "Accelergy: An Architecture-Level Energy Estimation Methodology for
  Accelerator Designs", ICCAD 2019 — `http://accelergy.mit.edu/paper.pdf`
  (mirror: `https://dspace.mit.edu/bitstream/handle/1721.1/122044/2019_iccad_accelergy_ilp.pdf`).
  Sections cited: §1.1 related work, §2 framework, §3.1 OO approach, §3.2 primitive, §3.3 compound
  (incl. Examples 1–2), §3.4 arch + action counts, §4 DNN action types + Table 1, §5.1–5.4
  validation.
- MICRO-52 tutorial slides — `https://accelergy.mit.edu/micro52/02_accelergy.pdf` (slides 13–16:
  the Timeloop/Accelergy block diagram).
- Project site — `https://accelergy.mit.edu/`

Source repos (all read at `master`/`main`)
- `https://github.com/Accelergy-Project/accelergy` — `README.md`, `accelergy/energy_calculator.py`,
  `accelergy/plug_in_interface/interface.py`, `accelergy/plug_in_interface/query_plug_ins.py`,
  `share/estimation_plug_ins/dummy_tables/dummy_table.py`,
  `test/tests/basic/data/action_counts.yaml`, `test/tests/action_area_share/inputs/components.yaml`
- `https://github.com/Accelergy-Project/timeloop-accelergy-exercises` —
  `workspace/cheatsheets/3_architecture.yaml`, `workspace/cheatsheets/4_compound_component.yaml`,
  `workspace/example_designs/example_designs/_components/smartbuffer_SRAM.yaml`,
  `workspace/example_designs/example_designs/simple_weight_stationary/{arch.yaml,
  ref_outputs/default_problem/timeloop-mapper.{ERT,ART}.yaml}`
- `https://github.com/Accelergy-Project/accelergy-cacti-plug-in` — `cacti_wrapper.py`
  (`_interp_technology`, `_interp_size`, `percent_accuracy_0_to_100`)
- `https://github.com/Accelergy-Project/accelergy-aladdin-plug-in` — README (40 nm, 5 ns default
  latency, `ALADDIN_ACCURACY = 70`)
- `https://github.com/Accelergy-Project/accelergy-table-based-plug-ins` — README (CSV format;
  deprecated)
- `https://github.com/Accelergy-Project/accelergy-library-plug-in` — README ("citations are required
  for all entries")
- `https://github.com/Accelergy-Project/hwcomponents` and `.../hwcomponents-cacti` — successor
  packages; adds latency models
- `https://github.com/Accelergy-Project/timeloopfe` — `timeloopfe/v4/processors/required_actions.py`

Timeloop docs
- `https://timeloop.csail.mit.edu/v4/parsing_and_intermediate_files/energy-and-area-reference-tables`
- `https://timeloop.csail.mit.edu/v4/output-formats/stats`
- `https://timeloop.csail.mit.edu/v4/input-formats/variables` (`global_cycle_seconds`,
  `action_latency_cycles`, `technology`, `n_instances`)

Eyexam (adjacent, not part of Accelergy)
- Y.-H. Chen, T.-J. Yang, J. Emer, V. Sze, "Eyeriss v2", arXiv:1807.07928 — Eyexam is in the
  appendix: "a sequential analysis process that involves seven major steps… starts with the
  assumption that the architecture has infinite processing parallelism, storage capacity and data
  bandwidth", each step tightening a roofline bound (Step 1 layer shape, … Step 5 storage capacity,
  … through NoC bandwidth). **Eyexam is a performance/throughput analysis methodology and contains
  no energy model; it is the conceptual complement to Accelergy, not a component of it.** Of the
  three tools in this family, Eyexam is the one aimed at the question we actually have (where does
  performance get lost, step by step), and it is a pencil-and-paper methodology rather than
  software.
