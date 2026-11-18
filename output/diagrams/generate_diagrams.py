"""Render documentation diagrams; does not train models or change project code.

Run from any directory: python output/diagrams/generate_diagrams.py
Outputs: SVG, PNG, editable Mermaid sources, and an offline HTML gallery.
The diagrams describe the source snapshot reviewed on 18 November 2026.
"""
from pathlib import Path
from html import escape
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.path import Path as MplPath

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
COLORS = {
    "process": ("#edf3ff", "#4267a8"),
    "data": ("#eaf6f3", "#297b70"),
    "decision": ("#fff5dc", "#aa791e"),
    "warning": ("#fff0e9", "#b4532b"),
    "terminal": ("#17344b", "#17344b"),
}


class Diagram:
    def __init__(self, name, title, subtitle, height=9, direction="TB"):
        self.name, self.title, self.subtitle = name, title, subtitle
        self.height, self.direction = height, direction
        self.nodes, self.edges, self.notes = {}, [], []

    def box(self, key, x, y, w, h, label, kind="process"):
        self.nodes[key] = (x, y, w, h, label, kind)
        return self

    def edge(self, start, end, label="", ports="bt", via=(), dashed=False, label_at=None):
        self.edges.append((start, end, label, ports, via, dashed, label_at))
        return self

    def anchor(self, key, port):
        x, y, w, h, _, _ = self.nodes[key]
        return {"t": (x, y-h/2), "b": (x, y+h/2),
                "l": (x-w/2, y), "r": (x+w/2, y)}[port]

    def render(self):
        fig, ax = plt.subplots(figsize=(12, self.height))
        fig.patch.set_facecolor("#ffffff")
        ax.set(xlim=(0, 12), ylim=(self.height, 0))
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.text(.45, .38, self.title, fontsize=19, fontweight="bold", color="#17344b", va="top")
        ax.text(.45, .84, self.subtitle, fontsize=10.5, color="#526578", va="top")
        ax.plot([.45, 11.55], [1.2, 1.2], color="#dce5ed", lw=1)
        mermaid = [f"%% {self.title}", f"%% {self.subtitle}", f"flowchart {self.direction}"]
        text_checks = []
        for key, (x, y, w, h, label, kind) in self.nodes.items():
            fill, stroke = COLORS[kind]
            if kind == "decision":
                patch = Polygon([(x,y-h/2), (x+w/2,y), (x,y+h/2), (x-w/2,y)],
                                facecolor=fill, edgecolor=stroke, lw=1.3, zorder=3)
                delimiters = ('{"', '"}')
            else:
                patch = FancyBboxPatch((x-w/2,y-h/2), w, h,
                    boxstyle="round,pad=0,rounding_size=0.09", facecolor=fill,
                    edgecolor=stroke, lw=1.2, zorder=3)
                delimiters = ('(["', '"])') if kind == "terminal" else ('["', '"]')
            ax.add_patch(patch)
            t = ax.text(x, y, label, ha="center", va="center", fontsize=10.1,
                        linespacing=1.5, color="white" if kind == "terminal" else "#17344b", zorder=4)
            text_checks.append((key, t, x, y, w, h))
            mermaid_label = escape(label).replace("\n", "<br/>")
            mermaid.append(f"    {key}{delimiters[0]}{mermaid_label}{delimiters[1]}:::{kind}")
        for start, end, label, ports, via, dashed, label_at in self.edges:
            points = [self.anchor(start, ports[0]), *via, self.anchor(end, ports[1])]
            path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO]*(len(points)-1))
            arrow = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
                lw=1.3, color="#63788b", linestyle="--" if dashed else "-", zorder=2)
            ax.add_patch(arrow)
            if label:
                lx, ly = label_at or ((points[0][0]+points[-1][0])/2, (points[0][1]+points[-1][1])/2)
                ax.text(lx, ly, label, ha="center", va="center", fontsize=8.6,
                        color="#526578", bbox=dict(fc="white", ec="none", pad=2), zorder=5)
            connector = "-.->" if dashed else "-->"
            mermaid.append(f'    {start} {connector}' + (f'|"{escape(label)}"|' if label else "") + f" {end}")
        for kind, (fill, stroke) in COLORS.items():
            mermaid.append(f"    classDef {kind} fill:{fill},stroke:{stroke},color:" + ("#ffffff" if kind == "terminal" else "#17344b"))
        for i, note in enumerate(self.notes):
            ax.text(.45, self.height-.30-.27*(len(self.notes)-i-1), note,
                    fontsize=8.8, color="#526578", va="bottom")
            mermaid.append("%% " + note)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for key, t, x, y, w, h in text_checks:
            bbox = t.get_window_extent(renderer)
            a, b = ax.transData.transform((x-w/2, y-h/2)), ax.transData.transform((x+w/2, y+h/2))
            if bbox.width > abs(b[0]-a[0]) - 8 or bbox.height > abs(b[1]-a[1]) - 6:
                raise ValueError(f"Text does not fit {self.name}:{key}")
        fig.savefig(OUT / f"{self.name}.svg", facecolor="white")
        fig.savefig(OUT / f"{self.name}.png", dpi=180, facecolor="white")
        plt.close(fig)
        (OUT / f"{self.name}.mmd").write_text("\n".join(mermaid)+"\n", encoding="utf-8")


def ga_architecture():
    d = Diagram("ga_architecture", "01  Genetic algorithm | architecture",
                "GeneticAlgorithm.py · analytical optimization · lower fitness is better", 8.6)
    d.box("settings", 2.25, 2, 3.5, 1.0, "Bounds + config + RNG seed\nDesign genes: [W, L, h] in mm", "data")
    d.box("run", 6, 2, 3, 1.0, "GeneticAlgorithm.run()\nPopulation and convergence control", "terminal")
    d.box("result", 9.75, 2, 3.5, 1.0, "results\nDesign, score, history, time", "data")
    d.box("initial", 2.25, 4, 3.5, 1.3, "initializePopulation()\nUniform sampling within bounds\nRepair antenna geometry")
    d.box("next", 6, 4, 3, 1.3, "create_next_generation()\nUnique elites + offspring\nReplace duplicates randomly")
    d.box("physics", 9.75, 4.25, 3.5, 1.8, "PatchFormula.RMA\nAnalytical frequency + volume\nfitness() + geometryFix()\nW/L ≥ 1.2 and box bounds", "data")
    d.box("operators", 5, 6.25, 5.3, 1.3, "select_parent() → crossover() → mutate()\nTournament → arithmetic blend → Gaussian change\nClip genes, repair geometry, refresh RMA")
    d.edge("settings", "run", ports="rl").edge("run", "result", ports="rl")
    d.edge("run", "initial", ports="bl", via=((4.2,2.8),(.3,2.8),(.3,4)))
    d.edge("run", "next").edge("run", "physics", ports="br", via=((7.7,2.8),(11.7,2.8),(11.7,4.25)))
    d.edge("next", "operators")
    d.edge("operators", "physics", "repair / score", ports="rt", via=((7.85,6.25),(7.85,3.0),(9.75,3.0)), label_at=(8.7,3.0))
    d.notes = ["Fitness: J = W·L·h / 11250 + λ |f_analytical − 2.4|. Frequency is in GHz; volume is in mm³.",
               "Current config constructor forces λ = 1 and terminationCondition = 0.01, ignoring the supplied values.",
               "The __main__ driver repeats ten seeds, plots histories, and displays the lowest-fitness final result."]
    return d


def ga_flowchart():
    d = Diagram("ga_flowchart", "02  Genetic algorithm | flowchart",
                "GeneticAlgorithm.run() and create_next_generation() · current control flow", 13.9)
    d.box("start", 4.5, 1.75, 5.3, .65, "Start timer; create seeded GA instance", "terminal")
    d.box("initial", 4.5, 2.85, 5.3, .9, "Initialize population of RMA antenna objects\nRepair dimensions; calculate analytical f and volume")
    d.box("score0", 4.5, 4, 5.3, .85, "Evaluate initial population; record minimum score\nSet generation = 0 and relative change = ∞")
    d.box("elites", 4.5, 5.15, 5.3, .85, "Sort current population by fitness\nCopy unique elites into next population")
    d.box("breed", 4.5, 6.45, 5.3, 1.1, "Select two parents by tournament\nBlend genes with crossover probability\nMutate each gene with Gaussian noise")
    d.box("repair", 4.5, 7.85, 5.3, 1.0, "Clip bounds and repair W/L ≥ 1.2\nIf duplicate: try a fresh repaired random design\nAppend unseen children until population is full")
    d.box("full", 9.3, 7.85, 3.4, 1.35, "Population\nfull?", "decision")
    d.box("evaluate", 4.5, 9.3, 5.3, .9, "Evaluate new population; record its best design / score\nAppend history; increment generation")
    d.box("stop", 4.5, 10.9, 5.6, 1.7, "Generation ≥ generationsMin\nand relative change < 0.01?", "decision")
    d.box("done", 4.5, 12.45, 5.3, .8, "Return design, fitness, history, generations and time", "terminal")
    for a,b in [("start","initial"),("initial","score0"),("score0","elites"),("elites","breed"),("breed","repair")]:
        d.edge(a,b)
    d.edge("repair","full",ports="rl")
    d.edge("full","breed","No",ports="tr",via=((9.3,6.45),),label_at=(9.3,6.95))
    d.edge("full","evaluate","Yes",ports="br",via=((9.3,9.3),),label_at=(9.3,8.9))
    d.edge("evaluate","stop")
    d.edge("stop","elites","No",ports="ll",via=((.65,10.9),(.65,5.15)),label_at=(.65,9.1))
    d.edge("stop","done","Yes",label_at=(4.75,11.98))
    d.notes = ["Relative change = |J_current − J_previous| / |J_previous|; this is score stagnation, not frequency acceptance.",
               "There is no maximum-generation cap. Defaults shown by __main__: population 6, minimum 100 generations, 1 elite.",
               "Crossover probability 0.8; per-gene mutation probability 0.2; mutation scale = 0.1 × each dimension's range."]
    return d


def nn_architecture():
    d = Diagram("nn_architecture", "03  Neural network | architecture",
                "NeuralNetwork.py · 3 inputs → 32 tanh neurons → 1 linear output · 161 parameters", 10)
    d.box("input", 2.2, 2.15, 3.3, 1.1, "Design input\n[W_mm, L_mm, h_mm]\nPublic predict(): one design", "data")
    d.box("scale", 6, 2.15, 3.5, 1.1, "Standardize inputs\nX_s = (X − X_mean) / X_std\nTraining-set statistics only")
    d.box("hidden", 9.8, 2.15, 3.3, 1.1, "32-neuron hidden layer\nH = tanh(W_hidden X_sᵀ + B_hidden)\nH shape: (32, N)")
    d.box("linear", 9.8, 4.35, 3.3, 1.1, "1-neuron linear output\nY_s = (W_output H + B_output)ᵀ\nY_s shape: (N, 1)")
    d.box("inverse", 6, 4.35, 3.5, 1.1, "Undo target standardization\nY_GHz = Y_s × Y_std + Y_mean\nSaved training-set statistics")
    d.box("freq", 2.2, 4.35, 3.3, 1.1, "Predicted resonance frequency\npredict() returns a float in GHz\nUsed as a fixed PSO surrogate", "data")
    d.box("trainer", 3.55, 6.75, 6, 1.65, "Training: fit()\nTraining / validation split → standardized MSE\nManual backward() → Strong Wolfe line search\nRestore weights from the best validation epoch")
    d.box("weights", 9.25, 6.75, 4.3, 1.65, "Trainable weights and biases\nW_hidden: (32, 3); B_hidden: (32, 1)\nW_output: (1, 32); B_output: (1, 1)\n96 + 32 + 32 + 1 = 161")
    d.box("saved", 6, 8.55, 7.5, .8, "save() / load(): NPZ weights, scaling, settings and training histories", "data")
    d.edge("input","scale",ports="rl").edge("scale","hidden",ports="rl").edge("hidden","linear")
    d.edge("linear","inverse",ports="lr").edge("inverse","freq",ports="lr")
    d.edge("trainer","weights","updates",ports="rl",dashed=True,label_at=(6.8,6.4))
    d.edge("weights","linear",ports="tt",via=((11.65,5.55),(11.65,3.6),(9.8,3.6)),dashed=True)
    d.edge("weights","hidden",ports="rt",via=((11.75,6.75),(11.75,1.4),(9.8,1.4)),dashed=True)
    d.edge("trainer","saved").edge("weights","saved")
    d.notes=["N is the number of samples. forward() accepts standardized arrays (N, 3); predict() accepts a single raw-mm design.",
             "hiddenSize is configurable; 32 is the value in __main__. Array operations use CuPy if available, otherwise NumPy."]
    return d


def nn_flowchart():
    d = Diagram("nn_flowchart", "04  Neural network | training flowchart",
                "Default __main__ path: full-batch training with Strong Wolfe line search", 14.8)
    d.box("data", 4.4, 1.85, 5.6, .85, "Load antennaSample_training.npz\nUse saved training, validation and test indices", "data")
    d.box("init", 4.4, 3.0, 5.6, .9, "Initialize 3 → 32 → 1 network (seed 1)\nXavier weights and zero biases")
    d.box("scale", 4.4, 4.15, 5.6, .9, "Fit X and Y mean / standard deviation on training only\nTransform training and validation data")
    d.box("grad", 4.4, 5.35, 5.6, .9, "Forward pass on the full training batch\nManual backpropagation of standardized MSE")
    d.box("line", 4.4, 6.6, 5.6, 1.0, "Search along d = −gradient; initial step α = 1\nDouble trial α to bracket; bisect (zoom) as needed\nAccept sufficient decrease AND Strong Wolfe curvature")
    d.box("accepted", 9.4, 6.6, 3.3, 1.5, "Nonzero step\naccepted?", "decision")
    d.box("stall", 9.4, 8.5, 3.3, .95, "Flag stalled training\nOn RuntimeError: restore\npre-search parameters", "warning")
    d.box("metrics", 4.4, 8.5, 5.6, 1.0, "Evaluate current training and validation MSE (GHz²)\nAppend loss histories; if validation improves:\nsave best parameters and epoch")
    d.box("stop", 4.4, 10.25, 5.8, 1.8, "Target RMSE reached,\npatience exhausted, stalled,\nor epoch limit reached?", "decision")
    d.box("restore", 4.4, 11.85, 5.6, .75, "Restore the best validation parameters; return history")
    d.box("test", 4.4, 13.05, 5.6, 1.0, "Save NPZ model and training plot; reload model\nEvaluate held-out test designs against RMA\nSave comparison plots, predictions CSV and metrics JSON", "data")
    for a,b in [("data","init"),("init","scale"),("scale","grad"),("grad","line")]:
        d.edge(a,b)
    d.edge("line","accepted",ports="rl")
    d.edge("accepted","metrics","Yes",ports="bt",via=((9.4,7.55),(4.4,7.55)),label_at=(6.7,7.55))
    d.edge("accepted","stall","No",ports="rr",via=((11.5,6.6),(11.5,8.5)),label_at=(11.5,7.6))
    d.edge("stall","metrics",ports="lr")
    d.edge("metrics","stop")
    d.edge("stop","grad","No: next epoch",ports="ll",via=((.55,10.25),(.55,5.35)),label_at=(.7,7.05))
    d.edge("stop","restore","Yes",label_at=(4.65,11.35)).edge("restore","test")
    d.notes=["__main__: at most 50,000 epochs; best validation RMSE < 10 kHz; patience = 2,000 epochs without improvement.",
             "fit() also supports minibatches and fixed-rate gradient updates when line_search=False; these are not the main path.",
             "The test split is evaluated after training. Line search itself applies the accepted update; there is no second update."]
    return d


def pso_architecture():
    d = Diagram("pso_architecture", "05  Particle swarm | architecture",
                "PSO.py · intended surrogate integration; dashed links mark connections needing repair", 11.0)
    d.box("models", 2.3, 2.1, 3.6, 1.0, "Saved NN and GP models\nNeuralNetwork.load()\nGPRskeleton.GaussianProcessRegressor.load()", "data")
    d.box("driver", 6.2, 2.1, 3.3, 1.0, "main() → compare_models()\nNN and GP; seeds 1–5\nSame bounds and score settings")
    d.box("objective", 10, 2.1, 3.1, 1.0, "make_objective()\nVolume + predicted-f penalty\nCalls model.predict(design)")
    d.box("swarm", 6.2, 4.4, 4.5, 1.5, "PSO class\ninitialize_swarm() / evaluate()\nupdate_velocity() / move_particle()\nrun(): personal and global bests")
    d.box("repair", 1.95, 4.4, 3.1, 1.5, "repair_design()\nClip bounds and use\nRMA.geometryFix()\nCurrently single-design only", "warning")
    d.box("result", 10.2, 4.4, 2.6, 1.5, "PSO.Result\nBest design / score\nPosition + score histories\nEvaluations + time", "data")
    d.box("verify", 6.2, 6.7, 4.5, 1.4, "verify_results(design, model)\nSurrogate prediction vs analytical RMA\nTarget error + prediction error\nGeometry and frequency tolerance", "data")
    d.box("report", 10.2, 6.7, 2.6, 1.4, "Comparison outputs\nHistory CSVs\ncomparison.json\nSuccess rate + time", "data")
    d.box("issues", 6, 9.05, 10.9, 1.7, "Current integration blockers\n1. compare_models passes object instead of objective to PSO.\n2. initialize_swarm and move_particle pass (swarm_size, 3) arrays to a single-design repair.\n3. evaluate builds an object array from a generator; it never calls self.objective.\n4. Verification and saving sit outside the seed loop, so only the final seed would be recorded.", "warning")
    d.edge("models","driver",ports="rl").edge("driver","objective",ports="rl")
    d.edge("driver","swarm","construct + run",label_at=(6.2,3.25))
    d.edge("objective","swarm","objective callback",ports="bl",via=((10,2.9),(3.7,2.9),(3.7,4.4)),dashed=True,label_at=(8.7,2.9))
    d.edge("swarm","repair","repair",ports="ll",via=((3.8,5.55),(.25,5.55),(.25,4.4)),dashed=True,label_at=(1.9,5.55))
    d.edge("swarm","result",ports="rl").edge("result","verify",ports="bl",via=((10.2,5.7),(3.7,5.7),(3.7,6.7)))
    d.edge("verify","report",ports="rl")
    d.notes=["Intended score: J = W·L·h / 11250 + λ |f_predicted − 2.4|. main() sets λ = 1 and analytical tolerance = 0.01 GHz.",
             "The network is fixed during search. Analytical target error and surrogate prediction error are separate quantities."]
    return d


def pso_flowchart():
    d = Diagram("pso_flowchart", "06  Particle swarm | flowchart",
                "Intended search loop in run(); default execution currently fails at swarm repair", 14.3)
    d.box("start", 4.3, 1.85, 5.4, .75, "Start timer and seeded RNG\nCompute velocity limits from bounds", "terminal")
    d.box("init", 4.3, 3.05, 5.4, 1.0, "Sample positions and velocities\nRepair positions; evaluate objective per particle\nInitialize personal bests and global best", "warning")
    d.box("blocked", 9.5, 3.05, 3.5, 1.25, "Current runtime blocker\n40 × 3 swarm passed to\nrepair_design() → RMA(*design)\nTypeError", "warning")
    d.box("continue", 4.3, 5.0, 5.8, 2.0, "iteration < max\nand (iteration < min\nor relative improvement > target)?", "decision")
    d.box("velocity", 4.3, 6.85, 5.4, 1.0, "Update velocity with inertia, personal and global best\nv ← ωv + c₁r₁(pbest − x) + c₂r₂(gbest − x)\nClip each component to [−vmax, +vmax]")
    d.box("move", 4.3, 8.15, 5.4, .9, "Move particles: x ← x + v\nRepair geometry; evaluate fixed surrogate objective")
    d.box("best", 4.3, 9.45, 5.4, 1.0, "Update each personal best if its score improves\nUpdate global best from all personal bests\nAppend global best score and position histories")
    d.box("progress", 4.3, 10.8, 5.4, .9, "Compute relative score improvement; iteration += 1\n(J_previous − J_best) / max(|J_previous|, 10⁻¹²)")
    d.box("return", 9.5, 5.0, 3.5, 1.0, "Return PSO.Result\nBest design, histories,\nevaluation count and time", "terminal")
    d.box("verify", 9.5, 7.0, 3.5, 1.5, "Driver: verify_results()\nCompute analytical frequency\nCheck geometry and target error\nCompare surrogate prediction", "data")
    d.box("report", 9.5, 9.15, 3.5, 1.1, "Driver: record histories / metrics\nSummarize NN vs GP\nSave comparison.json", "data")
    d.box("note", 6, 12.45, 10.9, .9, "The continuation below initialization describes the intended loop after fixing the listed connections.\nCurrent compare_models() also records only the final seed because its reporting block is outside the seed loop.", "warning")
    d.edge("start","init").edge("init","blocked","actual",ports="rl")
    d.edge("init","continue","after repairs",dashed=True,label_at=(4.3,3.8))
    d.edge("continue","velocity","Yes",label_at=(4.6,6.18))
    d.edge("continue","return","No",ports="rl")
    d.edge("velocity","move").edge("move","best").edge("best","progress")
    d.edge("progress","continue",ports="ll",via=((.5,10.8),(.5,5)))
    d.edge("return","verify").edge("verify","report")
    d.notes=["Defaults: 40 particles; min 150 / max 1,000 iterations; improvement target 0.01; ω = 0.7; c₁ = c₂ = 1.6.",
             "vmax = 0.2 × (upper − lower). r₁ and r₂ are independent random arrays shaped (swarm_size, 3).",
             "Stopping uses score improvement; success is checked separately using analytical |f − 2.4| ≤ 0.01 GHz and geometry."]
    return d


SECTIONS = [
    ("GeneticAlgorithm.py", "Direct analytical optimization", [ga_architecture, ga_flowchart]),
    ("NeuralNetwork.py", "Frequency surrogate: architecture and training", [nn_architecture, nn_flowchart]),
    ("PSO.py", "Surrogate optimization: intended workflow and current blockers", [pso_architecture, pso_flowchart]),
]


def build_gallery():
    sections = []
    for file, subtitle, builders in SECTIONS:
        figures = []
        for builder in builders:
            d = builder()
            d.render()
            figures.append(f'''<article><h3>{escape(d.title)}</h3>
<p class="downloads"><a href="{d.name}.svg">SVG</a> · <a href="{d.name}.png">PNG</a> · <a href="{d.name}.mmd">Editable Mermaid</a></p>
<a href="{d.name}.svg"><img src="{d.name}.svg" alt="{escape(d.title)}"></a></article>''')
        sections.append(f'<section id="{file[:-3]}"><h2><a href="../../src/{file}">{file}</a></h2><p>{subtitle}</p>' + "".join(figures) + "</section>")
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>EE5606 algorithm diagrams</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f8;color:#17344b;font:16px/1.6 system-ui,Segoe UI,sans-serif}
main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2;margin-bottom:12px}
h2{margin:48px 0 6px}h3{margin:0}a{color:#245a91}nav{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}
article{background:white;border:1px solid #dbe5ec;border-radius:12px;margin:22px 0;padding:20px}
article img{width:100%;height:auto;display:block}.downloads{font-size:14px;margin:6px 0 16px}
.note{border-left:4px solid #b4532b;background:#fff0e9;padding:16px 20px}.muted{color:#526578;font-size:14px}
table{width:100%;border-collapse:collapse;background:white;margin:18px 0}th,td{text-align:left;vertical-align:top;padding:12px;border:1px solid #dbe5ec}
code{font-size:.9em} @media print{body{background:white}main{padding:0}nav,.downloads{display:none}article{break-before:page;border:0;padding:0}article img{max-height:90vh;object-fit:contain}h2{break-before:page}}
</style></head><body><main>
<p class="muted">EE5606 · RECTANGULAR MICROSTRIP PATCH ANTENNA · SOURCE SNAPSHOT 18 NOV 2026</p>
<h1>Algorithm architectures &amp; flowcharts</h1>
<p>Six diagrams explain the current GA, neural network and PSO files. Architecture diagrams show components and data;
flowcharts show execution order, decisions and loops. Use SVG for scalable report/slide graphics, PNG for easy insertion,
or edit the Mermaid (.mmd) sources in a Mermaid-compatible editor. This gallery works offline.</p>
<nav><a href="#GeneticAlgorithm">Genetic algorithm</a><a href="#NeuralNetwork">Neural network</a><a href="#PSO">Particle swarm</a><a href="#notes">Code notes</a></nav>
<div class="note"><strong>Implementation status:</strong> GA and NN diagrams describe their current code.
PSO diagrams label the intended continuation and identify incomplete connections. The source algorithms were not modified.</div>
<p>All design vectors use <code>[W_mm, L_mm, h_mm]</code>. Bounds are W = 25–50 mm, L = 25–45 mm,
h = 1–5 mm, with W/L ≥ 1.2. Frequencies are in GHz; the target is 2.4 GHz.</p>
''' + "".join(sections) + '''
<section id="notes"><h2>Code notes and interpretation</h2>
<table><thead><tr><th>Source location</th><th>What the diagrams reflect</th></tr></thead><tbody>
<tr><td>GeneticAlgorithm.py:44–45</td><td>The config constructor assigns terminationCondition = 0.01 and frequencyPenalty = 1 regardless of its arguments. The driver's requested penalty 10000 is therefore not applied.</td></tr>
<tr><td>GeneticAlgorithm.py:169–179</td><td>Stopping checks the change between consecutive population-best scores after the minimum generation count. There is no maximum-generation cap or frequency-feasibility stopping check.</td></tr>
<tr><td>NeuralNetwork.py:67–98, 271–275</td><td>The class accepts a configurable hidden width; the driver uses 32. Public predict() takes one design in millimetres, applies saved scaling, then returns one frequency in GHz.</td></tr>
<tr><td>NeuralNetwork.py:129–152, 225–241</td><td>Strong Wolfe search evaluates trial weights along the negative gradient and leaves the accepted parameters in place. Do not add a second gradient update on this path. A RuntimeError restores the pre-search parameters and flags training as stalled; other exceptions propagate.</td></tr>
<tr><td>NeuralNetwork.py:243–269</td><td>Training records losses in GHz² and returns the parameters from the best validation epoch. The test split is used only for evaluation after training.</td></tr>
<tr><td>PSO.py:65–74, 85–86</td><td>initialize_swarm() passes a (40, 3) array to a repair function expecting one (3,) design. A focused check reproduced TypeError at RMA(*design). move_particle() uses the same non-batched repair callback. evaluate() returns a zero-dimensional object array containing a generator rather than numeric scores; it never calls self.objective.</td></tr>
<tr><td>PSO.py:225</td><td>The optimizer receives Python's built-in object rather than the objective closure created by make_objective(). The dashed objective link marks the intended connection.</td></tr>
<tr><td>PSO.py:231–292</td><td>Verification, run recording and history saving are outside the inner seed loop. After other blockers are fixed, this placement would retain only the final seed for each model, not all five runs.</td></tr>
</tbody></table>
<p><strong>Score versus success.</strong> Both score formulas are V/11250 + λ|f − 2.4|. GA uses analytical f;
the PSO objective is designed to use a frozen surrogate's f. PSO verification separately reports
analytical target error |f_analytical − 2.4| and surrogate prediction error |f_predicted − f_analytical|.
A converged score is not evidence of frequency feasibility.</p>
<p class="muted">These are code diagrams, not experimental results. No model training or optimization experiment was run to generate them.
PNG/SVG text bounds are checked by the generator. <a href="generate_diagrams.py">Regeneration script</a> · <a href="source_snapshot.json">Source hashes</a></p>
</section></main></body></html>'''
    (OUT / "index.html").write_text(html, encoding="utf-8")
    snapshot = {"reviewed_date": "2026-11-18", "sources": {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "src" / name for name in ("GeneticAlgorithm.py", "NeuralNetwork.py", "PSO.py", "PatchFormula.py")]
    }}
    (OUT / "source_snapshot.json").write_text(json.dumps(snapshot, indent=2)+"\n", encoding="utf-8")
    print(f"Created six diagrams (SVG, PNG, Mermaid) and gallery: {OUT / 'index.html'}")


if __name__ == "__main__":
    build_gallery()
