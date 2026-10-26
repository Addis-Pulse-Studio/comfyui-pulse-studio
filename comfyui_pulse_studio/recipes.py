"""Sampling recipes: the steps, sampler, schedule and guidance one checkpoint wants.

Until these existed a PulseSlate wrote one set of sampler widgets into every
window of the film. That was fine while every window sampled the same checkpoint,
and it stopped being fine when the distilled ones arrived, because they do not
agree with each other:

  * FastH3-Omni (a pruned ref2va PDD student) wants 8 Euler steps on the
    `simple` schedule, and VSA attention at 10% keep -- its 32 output heads are
    chosen from the sampler's own sigmas, so the schedule is part of the model.
  * TaoMate (a LoRA on the full fl2va checkpoint) wants 3 Euler steps on a
    hand-written schedule -- states 0/16/33/49 of a 50-step shift-12 one.
  * HyperFlow wants exactly its own 8-step grid, which its pack registers as the
    `hyperflow` scheduler.

A film that opens on FastH3-Omni and carries on in TaoMate needs two of those at
once. A recipe is one of them, and PulseSlate takes one per branch.

Every number here was read from the model's own release, not inferred:

  * FastH3-Omni: `fastvideo_inference.json` in FastVideo/FastVideo-FastH3-Omni-8-
    Step-V1 (8 steps, guidance 1.0), and the official ComfyUI template
    `video_fasth3_ominiref_r2v.json` (euler, simple, 8, VSA keep 10).
  * TaoMate: the schedule its `infer.py` keeps, also what the community ComfyUI
    conversions ship with -- `1.0, 0.961165, 0.853333, 0.0`.
  * HyperFlow: the `hyperflow_sigmas` header of the released adapter, served by
    ComfyUI-HyperFlow's scheduler, which refuses anything but 8 steps.

Pure stdlib -- the node layer reads the model's traits (it needs torch for that)
and this module only decides what they mean.
"""

from .constants import BRANCH_FL2VA, BRANCH_REF2VA

__all__ = [
    "RECIPE_PRESETS",
    "RECIPE_NAMES",
    "RECIPE_CUSTOM",
    "TAOMATE_3STEP_SIGMAS",
    "RecipeError",
    "make_recipe",
    "parse_sigmas",
    "recipe_problems",
    "recipe_warnings",
    "sampling_for_branch",
]

#: The three intervals TaoMate's FL2VA LoRA was trained to take.
TAOMATE_3STEP_SIGMAS = (1.0, 0.961165, 0.853333, 0.0)

RECIPE_CUSTOM = "custom"

#: Name -> settings. Order is the order of the PulseRecipe combo.
RECIPE_PRESETS = {
    "fasth3_omni_8step": {
        "steps": 8, "sampler": "euler", "scheduler": "simple", "cfg": 1.0, "sigmas": None,
        "branch": BRANCH_REF2VA,
        "about": "FastH3-Omni 8-step (ref2va). Patch the model with Model Attention "
                 "Backend -> Model Sparse Attention (vsa, keep 10, exact_kv_and_rows).",
    },
    "taomate_fl2va_3step": {
        "steps": 3, "sampler": "euler", "scheduler": "simple", "cfg": 1.0,
        "sigmas": list(TAOMATE_3STEP_SIGMAS), "branch": BRANCH_FL2VA,
        "about": "TaoMate FL2VA 3-step: the full fl2va checkpoint with the TaoMate "
                 "LoRA at 1.0, on its own three-interval schedule.",
    },
    "hyperflow_8step": {
        "steps": 8, "sampler": "euler", "scheduler": "hyperflow", "cfg": 1.0, "sigmas": None,
        "branch": None,
        "about": "HyperFlow 8-step, on either branch. Needs the ComfyUI-HyperFlow pack "
                 "and its LoRA loader on the (non-pruned) model.",
    },
    "h3_base": {
        "steps": 20, "sampler": "res_multistep", "scheduler": "simple", "cfg": 1.0,
        "sigmas": None, "branch": None,
        "about": "The undistilled checkpoints, at PulseSlate's own defaults.",
    },
}
RECIPE_NAMES = tuple(RECIPE_PRESETS) + (RECIPE_CUSTOM,)


class RecipeError(ValueError):
    """A recipe that cannot be sampled as written."""


def parse_sigmas(text):
    """A comma- or space-separated sigma list, or None for an empty one.

    Raises RecipeError on anything that is not a number, rather than sampling a
    schedule that silently dropped a value.
    """
    raw = (text or "").replace(",", " ").split()
    if not raw:
        return None
    values = []
    for item in raw:
        try:
            values.append(float(item))
        except ValueError:
            raise RecipeError("sigma %r is not a number" % (item,))
    return values


def make_recipe(name, steps=20, sampler="res_multistep", scheduler="simple", cfg=1.0,
                sigmas=""):
    """The recipe dict PulseRecipe emits. A preset ignores the custom widgets."""
    if name in RECIPE_PRESETS:
        preset = RECIPE_PRESETS[name]
        recipe = {k: preset[k] for k in ("steps", "sampler", "scheduler", "cfg")}
        recipe["sigmas"] = list(preset["sigmas"]) if preset["sigmas"] else None
    elif name == RECIPE_CUSTOM:
        recipe = {"steps": int(steps), "sampler": sampler, "scheduler": scheduler,
                  "cfg": float(cfg), "sigmas": parse_sigmas(sigmas)}
    else:
        raise RecipeError("recipe must be one of %s, got %r" % (", ".join(RECIPE_NAMES), name))
    recipe["name"] = name
    if recipe["sigmas"]:
        # A hand-written schedule fixes the step count; the widget cannot disagree.
        recipe["steps"] = len(recipe["sigmas"]) - 1
    return recipe


def recipe_problems(recipe):
    """Why this recipe cannot be sampled at all. Fatal, so caught at compile time.

    Every one of these used to surface as an exception inside the first window's
    sampler -- after the text encoder, the checkpoint and the references had all
    loaded, and after a dry run had said everything was fine.
    """
    problems = []
    label = recipe.get("name") or "sampler settings"
    sigmas = recipe.get("sigmas")
    if sigmas:
        if len(sigmas) < 2:
            problems.append("%s: a sigma schedule needs at least two values" % label)
        elif any(b > a for a, b in zip(sigmas, sigmas[1:])):
            problems.append("%s: sigmas must run downward, got %s"
                            % (label, ", ".join("%g" % v for v in sigmas)))
        elif abs(sigmas[-1]) > 1e-9:
            problems.append("%s: the last sigma must be 0, got %g -- the sampler would "
                            "stop with noise still in the frame" % (label, sigmas[-1]))
        elif sigmas[0] > 1.0 + 1e-6:
            problems.append("%s: a flow model's first sigma is at most 1.0, got %g"
                            % (label, sigmas[0]))
    elif recipe.get("scheduler") == "hyperflow" and int(recipe.get("steps") or 0) != 8:
        problems.append(
            "%s: the hyperflow schedule is HyperFlow's fixed 8-step grid, so steps "
            "must be 8 (got %s)" % (label, recipe.get("steps")))
    if int(recipe.get("steps") or 0) < 1:
        problems.append("%s: steps must be at least 1" % label)
    return problems


def recipe_warnings(recipe, traits, branch):
    """What is off-recipe for this checkpoint. Warnings, never fatal.

    `traits` is what the node layer read off the model: `pdd_heads` (FastH3's
    stacked output heads), `curve_form` (a pruned checkpoint), `vsa` / `sparse`
    (core's sparse attention), `hyperflow` (its adapter), `lora` (any weight
    patch). Missing keys read as "not detected".
    """
    traits = traits or {}
    out = []
    label = "%s (%s)" % (recipe.get("name") or "sampler settings", branch)
    scheduler = recipe.get("scheduler")
    pdd = int(traits.get("pdd_heads") or 1) > 1

    if pdd:
        if int(recipe.get("steps") or 0) != 8 or recipe.get("sampler") != "euler":
            out.append(
                "%s: this checkpoint has %d stacked PDD output heads -- it is a "
                "FastH3 student, distilled for 8 Euler steps -- and is being sampled "
                "at %s steps with %s. Use the fasth3_omni_8step recipe."
                % (label, int(traits["pdd_heads"]), recipe.get("steps"),
                   recipe.get("sampler")))
        if not traits.get("vsa"):
            out.append(
                "%s: FastH3-Omni was distilled with VSA attention at 10%% keep, and no "
                "Model Sparse Attention (vsa) patch was found on it. Dense attention "
                "is off-recipe and slower: chain Model Attention Backend -> Model "
                "Sparse Attention (vsa, keep 10) before PulseSlate." % label)
    if scheduler == "hyperflow" and not traits.get("hyperflow"):
        out.append(
            "%s: the hyperflow schedule is set, but no HyperFlow adapter was found on "
            "the model. Its 8-step grid is only right for a model HyperFlow patched -- "
            "load the LoRA with HyperFlow LoRA Loader." % label)
    if traits.get("hyperflow") and scheduler != "hyperflow" and not recipe.get("sigmas"):
        out.append(
            "%s: the model carries HyperFlow, which is trained for its own 8-step "
            "grid, but the schedule is %r. Use the hyperflow_8step recipe."
            % (label, scheduler))
    if traits.get("hyperflow") and traits.get("curve_form"):
        out.append(
            "%s: HyperFlow is on a pruned (curve-form) checkpoint, which it was not "
            "trained against. Use the non-pruned checkpoints." % label)
    if recipe.get("name") == "taomate_fl2va_3step" and not traits.get("lora"):
        out.append(
            "%s: TaoMate's 3-step schedule is set, but no LoRA was found on the "
            "model. Without the TaoMate adapter three steps leave the frame "
            "unfinished -- load it with LoraLoaderModelOnly at 1.0." % label)
    if recipe.get("name") == "taomate_fl2va_3step" and branch != BRANCH_FL2VA:
        out.append(
            "%s: TaoMate is an fl2va LoRA, and this recipe is on the %s branch."
            % (label, branch))
    if recipe.get("name") == "fasth3_omni_8step" and branch != BRANCH_REF2VA:
        out.append(
            "%s: FastH3-Omni is a ref2va checkpoint, and this recipe is on the %s "
            "branch." % (label, branch))
    return out


def sampling_for_branch(branch, slate, recipe=None, recipe_fl2va=None):
    """The settings one window samples with.

    fl2va windows take `recipe_fl2va`, then `recipe`, then the slate's widgets;
    ref2va windows take `recipe`, then the widgets. Falling back to `recipe` on
    fl2va is what a film on one model family wants -- HyperFlow on both
    checkpoints is one recipe, wired once.
    """
    chosen = (recipe_fl2va or recipe) if branch == BRANCH_FL2VA else recipe
    if not chosen:
        return dict(slate, sigmas=None, name=None)
    return dict(chosen)
