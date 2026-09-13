"""How many arms a RoboTwin expert uses, read statically from its ``play_once``.

A one-arm competition may score only tasks one arm can do, and which those are is a property of
the expert's source. ``tasks.yml`` records it per task; this module re-derives it from the pinned
checkout so the table cannot drift. Each ``envs/<task>.py`` is parsed with :mod:`ast` — never
imported, so no simulator is needed — and the expert is read as ``play_once`` plus every method
and nested function it reaches, because several experts do their work in helpers.

The rule, from a read of every expert at the pinned commit:

- ``"2"``: both arms act. Either one ``self.move`` carries two arm motions at once, or two
  different arm identities act over the episode — a fixed ``"left"`` and a fixed ``"right"``, a
  chosen arm and its ``.opposite``, a chosen arm and a fixed one. Sending an arm
  ``back_to_origin`` is not acting: it is how an expert clears the other arm out of the way.
- ``"switching"``: one arm acts at a time, but which one is chosen per object: an arm built
  from a pose inside a loop, or in a helper the expert calls more than once.
- ``"1"``: one arm, chosen once from the scene or fixed.

An identity is what an arm expression resolves to through the expert's own assignments: the
literal ``"left"`` or ``"right"``, ``opposite(...)`` of another identity, or the text of the
expression that chooses it (``ArmTag("left" if x < 0 else "right")``). Two uses of the same
choice are one arm; a second, different choice is another.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

ONE = "1"
SWITCHING = "switching"
TWO = "2"
ARMS = (ONE, SWITCHING, TWO)

LABELS = {ONE: "one arm", SWITCHING: "switching arms", TWO: "two arms"}

# Upstream files in `envs/` that are not tasks.
NON_TASK_STEMS = frozenset({"__init__", "_base_task", "_GLOBAL_CONFIGS"})

# Base_Task methods that produce an arm motion, with the position of their arm-tag argument.
_MOTIONS = {
    "grasp_actor": 1,
    "place_actor": 1,
    "move_by_displacement": 0,
    "move_to_pose": 0,
    "open_gripper": 0,
    "close_gripper": 0,
    "back_to_origin": 0,
}
_RETREAT = "back_to_origin"
_ARM_KEYWORD = "arm_tag"
_TAG_CLASS = "ArmTag"
_ACTION_CLASS = "Action"
_MOVE = "move"
_LITERALS = frozenset({"left", "right"})
_LOOPS = (
    ast.For,
    ast.While,
    ast.AsyncFor,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


class ArmsError(ValueError):
    """The expert cannot be read: no ``play_once``, or one that moves no arm."""


@dataclass(frozen=True)
class Verdict:
    arms: str
    evidence: str


def classify_arms(envs_dir: Path) -> dict[str, str]:
    """``arms`` for every task in a RoboTwin ``envs/`` directory, by task name."""
    return {task: verdict.arms for task, verdict in classify_all(envs_dir).items()}


def classify_all(envs_dir: Path) -> dict[str, Verdict]:
    envs_dir = Path(envs_dir)
    if not envs_dir.is_dir():
        raise ArmsError(f"no RoboTwin envs directory at {envs_dir}; init the submodule")
    verdicts = {}
    for path in sorted(envs_dir.glob("*.py")):
        if path.stem in NON_TASK_STEMS:
            continue
        verdicts[path.stem] = classify_source(path.read_text(encoding="utf-8"), path.stem)
    return verdicts


def classify_source(source: str, name: str = "<task>") -> Verdict:
    """Classify one task module's source."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and any(
            isinstance(item, ast.FunctionDef) and item.name == "play_once" for item in node.body
        ):
            return _Expert(node, name).verdict()
    raise ArmsError(f"{name}: no class defines play_once")


# --- the reader -------------------------------------------------------------------------------


@dataclass
class _Unit:
    """One function the expert runs: a method or a nested def, and how often it is entered."""

    node: ast.FunctionDef
    repeated: bool = False
    calls: int = 0


@dataclass(frozen=True)
class _Use:
    """An arm expression in an acting position."""

    identities: frozenset[str]
    line: int


class _Expert:
    def __init__(self, cls: ast.ClassDef, name: str) -> None:
        self.name = name
        self.methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.units: dict[str, _Unit] = {}
        self.assignments: dict[str, list[ast.expr]] = {}
        self._collect("play_once", repeated=False)
        for unit in self.units.values():
            self._gather_assignments(unit.node)

    # -- collection -------------------------------------------------------------------------

    def _collect(self, name: str, repeated: bool) -> None:
        """Enter `name` (a method) once more; the first entry also walks it for callees."""
        unit = self.units.get(name)
        if unit is not None:
            unit.calls += 1
            unit.repeated = unit.repeated or repeated or unit.calls > 1
            if unit.repeated:
                self._propagate(unit)
            return
        unit = self.units[name] = _Unit(self.methods[name], repeated=repeated, calls=1)
        self._walk_calls(unit)

    def _walk_calls(self, unit: _Unit) -> None:
        """Register every helper `unit` calls, and whether it is called from a loop."""
        for node, _ in _iter_body(unit.node):
            if isinstance(node, ast.FunctionDef):
                # A nested def is its own unit; its call sites decide whether it repeats.
                self.methods.setdefault(node.name, node)
        for node, in_loop in _iter_body(unit.node):
            if not isinstance(node, ast.Call):
                continue
            callee = _self_method(node) or _plain_name(node)
            if callee is None or callee not in self.methods or callee == unit.node.name:
                continue
            if callee in _MOTIONS or callee == _MOVE:
                continue
            self._collect(callee, repeated=unit.repeated or in_loop)

    def _propagate(self, unit: _Unit) -> None:
        """A helper entered more than once repeats everything it calls."""
        for node, _ in _iter_body(unit.node):
            if isinstance(node, ast.Call):
                callee = _self_method(node) or _plain_name(node)
                if callee in self.units and not self.units[callee].repeated:
                    self.units[callee].repeated = True
                    self._propagate(self.units[callee])

    def _gather_assignments(self, function: ast.FunctionDef) -> None:
        for node, _ in _iter_body(function):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                pairs = [(target, node.value)]
                if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    pairs = list(zip(target.elts, node.value.elts, strict=False))
                for name, value in pairs:
                    key = _key(name)
                    if key is not None:
                        self.assignments.setdefault(key, []).append(value)

    # -- identities -------------------------------------------------------------------------

    def identity(self, expr: ast.expr, seen: frozenset[str] = frozenset()) -> frozenset[str]:
        if isinstance(expr, ast.Constant):
            return frozenset({expr.value}) if expr.value in _LITERALS else frozenset()
        if isinstance(expr, ast.Attribute) and expr.attr == "opposite":
            return frozenset(f"opposite({i})" for i in self.identity(expr.value, seen))
        if isinstance(expr, ast.Call) and _plain_name(expr) == _TAG_CLASS and expr.args:
            inner = expr.args[0]
            if isinstance(inner, ast.Constant | ast.Name | ast.Attribute):
                return self.identity(inner, seen)
            return frozenset({f"chosen({ast.unparse(inner)})"})
        key = _key(expr)
        if key is not None:
            if key in seen:
                return frozenset()
            values = self.assignments.get(key)
            if not values:
                return frozenset({key})
            found: set[str] = set()
            for value in values:
                found |= self.identity(value, seen | {key})
            return frozenset(found)
        return frozenset({f"chosen({ast.unparse(expr)})"})

    # -- the rule ---------------------------------------------------------------------------

    def verdict(self) -> Verdict:
        acting: list[_Use] = []
        together: int | None = None
        choice_repeated: tuple[int, str] | None = None
        moves = 0
        for unit_name, unit in self.units.items():
            for node, in_loop in _iter_body(unit.node):
                if not isinstance(node, ast.Call):
                    if (
                        isinstance(node, ast.IfExp)
                        and _is_literal_choice(node)
                        and (unit.repeated or in_loop)
                        and choice_repeated is None
                    ):
                        choice_repeated = (node.lineno, unit_name)
                    continue
                method = _self_method(node)
                if method == _MOVE:
                    moves += 1
                    args = _move_args(node)
                    if together is None and sum(not _is_retreat(a) for a in args) >= 2:
                        together = node.lineno
                    for arg in args:
                        if isinstance(arg, ast.Tuple) and arg.elts:
                            acting.append(_Use(self.identity(arg.elts[0]), node.lineno))
                elif method in _MOTIONS and method != _RETREAT:
                    arm = _arm_argument(node, _MOTIONS[method])
                    if arm is not None:
                        acting.append(_Use(self.identity(arm), node.lineno))
                elif _plain_name(node) == _ACTION_CLASS and node.args:
                    acting.append(_Use(self.identity(node.args[0]), node.lineno))
                elif _plain_name(node) == _TAG_CLASS and node.args:
                    inner = node.args[0]
                    if (
                        not isinstance(inner, ast.Constant | ast.Name | ast.Attribute | ast.IfExp)
                        and (unit.repeated or in_loop)
                        and choice_repeated is None
                    ):
                        choice_repeated = (node.lineno, unit_name)
        if not moves:
            raise ArmsError(f"{self.name}: play_once moves no arm")

        identities: dict[str, int] = {}
        for use in acting:
            for identity in use.identities:
                identities.setdefault(identity, use.line)
        if together is not None:
            return Verdict(TWO, f"both arms move together at line {together}")
        if len(identities) >= 2:
            listed = ", ".join(f"{i} (line {line})" for i, line in sorted(identities.items()))
            return Verdict(TWO, f"different arms act: {listed}")
        if choice_repeated is not None:
            line, unit_name = choice_repeated
            where = f"in {unit_name}" if unit_name != "play_once" else "in a loop"
            calls = self.units[unit_name].calls
            detail = f"called {calls} times" if calls > 1 else "in a loop"
            return Verdict(SWITCHING, f"arm chosen per object at line {line} {where}, {detail}")
        if not identities:
            raise ArmsError(f"{self.name}: play_once moves an arm no motion names")
        ((identity, line),) = identities.items()
        return Verdict(ONE, f"one arm, {identity} (line {line})")


# --- ast helpers ------------------------------------------------------------------------------


def _iter_body(function: ast.FunctionDef):
    """Every node in `function`'s body with whether it sits in a loop; nested defs are yielded
    but not entered, so they can be read as units of their own."""
    stack: list[tuple[ast.AST, bool]] = [(child, False) for child in reversed(function.body)]
    while stack:
        node, in_loop = stack.pop()
        yield node, in_loop
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef):
            continue
        inside = in_loop or isinstance(node, _LOOPS)
        stack.extend((child, inside) for child in reversed(list(ast.iter_child_nodes(node))))


def _self_method(call: ast.Call) -> str | None:
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
    ):
        return func.attr
    return None


def _plain_name(call: ast.Call) -> str | None:
    return call.func.id if isinstance(call.func, ast.Name) else None


def _key(expr: ast.expr) -> str | None:
    """The assignment key of a name or a `self.<attr>`; None for anything else."""
    if isinstance(expr, ast.Name):
        return expr.id
    if (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
    ):
        return f"self.{expr.attr}"
    return None


def _move_args(call: ast.Call) -> list[ast.expr]:
    args = list(call.args)
    args.extend(
        kw.value for kw in call.keywords if kw.arg in ("actions_by_arm1", "actions_by_arm2")
    )
    return args


def _is_retreat(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Call) and _self_method(expr) == _RETREAT


def _arm_argument(call: ast.Call, position: int) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == _ARM_KEYWORD:
            return keyword.value
    return call.args[position] if len(call.args) > position else None


def _is_literal_choice(node: ast.IfExp) -> bool:
    return all(
        isinstance(branch, ast.Constant) and branch.value in _LITERALS
        for branch in (node.body, node.orelse)
    )
