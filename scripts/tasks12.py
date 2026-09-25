#!/usr/bin/env python3
"""Task-12 adapters shared by the SC and prompt-template drivers."""
from __future__ import annotations

import ast
import os
import re
import resource
import shutil
import string
import subprocess
import sys
import tempfile
from pathlib import Path


NEW_TASKS = ("math", "triviaqa", "drop", "cruxeval", "mbpp")

_LATEX_GROUPED_NUMBER_RE = re.compile(
    r"\d+(?:(?:\{,\}|\\[,;! ]|~)\d+)+"
)
_LATEX_THOUSANDS_SEPARATOR_RE = re.compile(r"\{,\}|\\[,;! ]|~")
_LATEX_SPACING_MACRO_RE = re.compile(r"(?<!\\)\\(?:[,;:!]|\s)")


def normalize_latex_thousands_separators(text: str) -> str:
    r"""Remove LaTeX thousands separators only from valid digit groupings.

    Every group after the first must contain exactly three digits. This keeps
    non-thousands content such as ``2{,}3`` unchanged.
    """
    def replace_grouped_number(match: re.Match[str]) -> str:
        groups = _LATEX_THOUSANDS_SEPARATOR_RE.split(match.group(0))
        if all(len(group) == 3 for group in groups[1:]):
            return "".join(groups)
        return match.group(0)

    return _LATEX_GROUPED_NUMBER_RE.sub(replace_grouped_number, str(text))


def require_nonempty_slice(instances: list[dict], limit: int, task: str) -> list[dict]:
    """Slice a task dataset and refuse to let an empty experiment succeed."""
    selected = instances[:limit]
    if not selected:
        raise SystemExit(
            f"FATAL: {task} produced no instances after slicing to n={limit}; "
            "refusing an empty run"
        )
    return selected


def normalize_squad(answer: str | None) -> str:
    """SQuAD normalization: lowercase, punctuation/articles removal, whitespace folding."""
    if answer is None:
        return ""
    text = str(answer).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def canonicalize_numeric_or_text(answer: str | None) -> str | None:
    """Use the v2 numeric bucket when possible, otherwise SQuAD normalization."""
    if answer is None:
        return None
    clean = normalize_latex_thousands_separators(answer)
    clean = clean.replace(",", "").strip().rstrip(".").strip()
    if not clean:
        return None
    try:
        value = round(float(clean), 3)
        if value == 0:
            value = 0.0
        return str(int(value)) if value.is_integer() else str(value)
    except (TypeError, ValueError):
        normalized = normalize_squad(clean)
        return normalized or None


_ROW_BREAK = ""


# Vendored verbatim from EleutherAI lm-evaluation-harness,
# path lm_eval/tasks/minerva_math/utils.py,
# commit 9fd734a220d1b7cabe626add57c96af1d2cca30f, MIT License
# (Copyright (c) 2020 EleutherAI), implementing Lewkowycz et al. (2022)
# appendix D; do not edit.
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    # Remove unit words while whitespace still delimits them, so that
    # "2\pi cm" loses "cm" and "\left" keeps its "ft" (word boundaries
    # cannot match inside a LaTeX command name)
    for expr in REMOVED_EXPRESSIONS:
        if expr.isalpha():
            final_answer = re.sub(rf"(?<!\\)\b{expr}\b", "", final_answer)

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        if expr.isalpha():
            # Units glued to their surroundings once whitespace is gone;
            # skip LaTeX command spans so "ft" cannot corrupt "\left".
            # Trade-off: a zero-separator <command><unit> run like "\picm"
            # survives as one token instead of being stripped by accident.
            segments = re.split(r"(\\[a-zA-Z]+)", final_answer)
            for i in range(0, len(segments), 2):
                segments[i] = segments[i].replace(expr, "")
            final_answer = "".join(segments)
        else:
            final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    # \sqrt[3]a -> \sqrt[3]{a}: canonicalize indexed roots with an unbraced
    # argument before the shorthand rule below, which must skip them entirely
    # or its [^ {] class would consume the "[" of the index and corrupt
    # \sqrt[3]{8} into \sqrt{[}3]{8}.
    final_answer = re.sub(r"(sqrt)(\[[^\]]*\])(\w)", "\\1\\2{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)(?!\[)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000: strip commas only when they are true
    # thousands-group separators. A bare tuple like "0,1" would otherwise
    # pass the old digit check and get fused into the garbage number "01"
    # (16 MATH test-set gold answers are bare tuples).
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", final_answer):
        final_answer = final_answer.replace(",", "")

    return final_answer


_EQ_PLACEHOLDER = "\uE001"
_SINGLE_VARIABLE_RE = re.compile(r"(?:[A-Za-z]|\\[A-Za-z]+)(?:_\{?[A-Za-z0-9]+\}?)?")


def _guard_equation(text: str) -> str:
    """Keep Minerva's right-of-'=' rule only for ``<variable> = <value>``.

    Minerva keeps whatever follows the last '='. That scores a plane equation
    ``... = 0`` as ``0`` and a two-root answer ``x = 1 or x = 5`` as ``5``.
    A single '=' with a lone variable on the left is reduced to its right side;
    every other '=' is protected from the split and restored afterwards.
    """
    if text.count("=") == 1:
        lhs, rhs = text.split("=")
        if _SINGLE_VARIABLE_RE.fullmatch(lhs.strip()):
            return rhs
    return text.replace("=", _EQ_PLACEHOLDER)


def _fix_fracs(string: str) -> str:
    """Hendrycks et al. (2021) MATH ``_fix_fracs``: ``\\frac12`` -> ``\\frac{1}{2}``."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr:
            return string
        if substr[0] == "{":
            new_str += substr
            continue
        if len(substr) < 2:
            return string
        a, b = substr[0], substr[1]
        if b != "{":
            new_str += "{" + a + "}{" + b + "}" + substr[2:]
        else:
            new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def canonicalize_math(answer: str | None) -> str | None:
    """Canonicalize a MATH-500 answer in this exact pipeline.

    0. Return ``None`` for ``None``.
    1. Normalize LaTeX thousands separators and strip outer whitespace.
    2. Protect LaTeX row breaks with ``_ROW_BREAK``.
    3. Remove spacing macros before Minerva normalization.
    3a. Map ``\\tfrac``/``\\dfrac`` to ``\\frac`` and brace shorthand fractions
        (Hendrycks ``_fix_fracs``), since Minerva's shorthand regex mangles
        ``\\frac9{19}``.
    3b. Reduce ``<variable> = <value>`` to its value and protect every other
        '=' from Minerva's last-'=' split (see ``_guard_equation``).
    4. Apply the vendored Minerva final-answer normalization, then restore '='.
    5. Apply Hendrycks MATH rules absent from Minerva.
    6. Remove spacing macros and whitespace, then a trailing period.
    7. Restore protected LaTeX row breaks.
    8. Canonicalize genuine numeric forms; otherwise return the string or ``None``.

    The pre-pass prevents Minerva's ``,\\!`` removal from also deleting a tuple
    comma; ``1,\\!000`` still becomes ``1000`` because the pre-pass leaves
    ``1,000`` for Minerva's thousands rule.
    """
    if answer is None:
        return None
    text = normalize_latex_thousands_separators(answer).strip()
    text = text.replace(r"\\", _ROW_BREAK)
    text = _LATEX_SPACING_MACRO_RE.sub("", text)
    text = text.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    text = _fix_fracs(text)
    # braced numerator with a one-character denominator: \frac{270}7 -> \frac{270}{7}
    text = re.sub(r"\\frac\{([^{}]*)\}([^{\s\\])", r"\\frac{\1}{\2}", text)
    text = _guard_equation(text)
    text = normalize_final_answer(text)
    text = text.replace(_EQ_PLACEHOLDER, "=")
    text = text.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = _LATEX_SPACING_MACRO_RE.sub("", text)
    text = re.sub(r"\s+", "", text).rstrip(".")
    text = text.replace(_ROW_BREAK, r"\\")
    numeric_text = (
        text.replace(",", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text)
        else text
    )
    try:
        float(numeric_text)
    except (TypeError, ValueError):
        return text or None
    return canonicalize_numeric_or_text(text)


def canonicalize_python_literal(answer: str | None) -> str | None:
    """Canonicalize a Python literal without executing it."""
    if answer is None:
        return None
    text = " ".join(str(answer).strip().split())
    if not text:
        return None
    try:
        return repr(ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return text


def canonicalize_trivia_answer(answer: str | None, item: dict) -> str | None:
    """Map any correct TriviaQA alias to the one canonical gold bucket."""
    pred = normalize_squad(answer)
    if not pred:
        return None
    gold = normalize_squad(item["answer"])
    aliases = item.get("metadata", {}).get("aliases", [])
    alias_buckets = {gold, *(normalize_squad(alias) for alias in aliases)}
    return gold if pred in alias_buckets else pred


def canonicalize_drop_answer(answer: str | None, item: dict) -> str | None:
    """Map every accepted DROP reference to the first canonical reference bucket."""
    pred = canonicalize_numeric_or_text(answer)
    if pred is None:
        return None
    references = [item["answer"], *item.get("metadata", {}).get("aliases", [])]
    canonical_references = list(dict.fromkeys(
        reference
        for reference in (canonicalize_numeric_or_text(value) for value in references)
        if reference is not None
    ))
    if not canonical_references:
        raise ValueError("DROP instance has no non-empty answer references")
    return canonical_references[0] if pred in canonical_references else pred


def canonicalize_for_instance(answer: str | None, item: dict, task: str) -> str | None:
    """Return the vote bucket for a non-code Task-12 instance."""
    if task == "math":
        return canonicalize_math(answer)
    if task == "triviaqa":
        return canonicalize_trivia_answer(answer, item)
    if task == "drop":
        return canonicalize_drop_answer(answer, item)
    if task == "cruxeval":
        return canonicalize_python_literal(answer)
    if task == "mbpp":
        raise ValueError("MBPP buckets require sandboxed per-assert execution")
    raise ValueError(f"Unknown Task-12 canonicalization task: {task}")


def extract_code_completion(text: str) -> str:
    """Extract the last fenced Python block, or retain the complete raw completion."""
    matches = re.findall(r"```python[ \t]*\r?\n(.*?)```", str(text), re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else str(text).strip()


def _sandbox_backend() -> str:
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bwrap"
    raise RuntimeError(
        "MBPP execution requires an OS sandbox with network isolation: "
        "sandbox-exec on macOS or bwrap on Linux. No safe backend is available."
    )


def _resource_limits(timeout_seconds: float) -> None:
    cpu_seconds = max(1, int(timeout_seconds) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _sandbox_command(script: Path, work_dir: Path) -> list[str]:
    backend = _sandbox_backend()
    python = Path(sys.executable).resolve()
    if backend == "sandbox-exec":
        escaped = str(work_dir).replace('"', r'\"')
        profile = (
            '(version 1) (deny default) '
            '(allow process-exec) (deny process-fork) '
            '(allow file-read*) '
            f'(allow file-write* (subpath "{escaped}")) '
            '(deny network*)'
        )
        return ["sandbox-exec", "-p", profile, str(python), "-I", str(script)]
    return [
        "bwrap", "--die-with-parent", "--unshare-all", "--unshare-net",
        "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
        "--bind", str(work_dir), str(work_dir), "--chdir", str(work_dir),
        str(python), "-I", str(script),
    ]


# None until the preflight has run in this process; True once it has passed.
_SANDBOX_PREFLIGHT_OK: bool | None = None


def _preflight_sandbox(timeout_seconds: float) -> None:
    """Prove the sandbox can run anything at all, once per process.

    Without this, a sandbox that cannot start scores every completion "F", which
    is indistinguishable from a model that answers everything wrong. That failure
    was observed on a compute node with max_user_namespaces=0, where a correct
    completion scored "FF" and the run looked like a real zero.
    """
    global _SANDBOX_PREFLIGHT_OK
    if _SANDBOX_PREFLIGHT_OK:
        return
    with tempfile.TemporaryDirectory(prefix="diversity_combining-sandbox-preflight-") as tmp:
        work_dir = Path(tmp).resolve()
        script = work_dir / "preflight.py"
        sentinel = work_dir / "preflight-passed"
        script.write_text(
            "from pathlib import Path as _P\n"
            f"_P({str(sentinel)!r}).write_text('ok', encoding='utf-8')\n",
            encoding="utf-8",
        )
        try:
            result = subprocess.run(
                _sandbox_command(script, work_dir),
                cwd=work_dir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=max(timeout_seconds, 10.0),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "MBPP sandbox preflight timed out; the sandbox cannot run a "
                "trivial script, so no MBPP score from this process is meaningful."
            ) from exc
        if result.returncode != 0 or not sentinel.is_file():
            detail = (result.stderr or b"").decode("utf-8", "replace").strip()
            raise RuntimeError(
                "MBPP sandbox preflight failed, so every completion would score "
                "'F' regardless of its correctness. Sandbox said: "
                f"{detail or '<no stderr>'}"
            )
    _SANDBOX_PREFLIGHT_OK = True


def evaluate_mbpp_completion(
    completion: str,
    assertions: list[str],
    test_imports: list[str] | None = None,
    timeout_seconds: float = 10.0,
) -> str:
    """Return P/F per assertion after isolated, no-network subprocess execution."""
    if not assertions:
        raise ValueError("MBPP instance has no assertions")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    _sandbox_backend()  # fail before partially evaluating an instance
    _preflight_sandbox(timeout_seconds)
    code = extract_code_completion(completion)
    imports = "\n".join(test_imports or [])
    outcomes: list[str] = []
    deterministic_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "LC_ALL": "C",
        "LANG": "C",
    }
    for assertion in assertions:
        with tempfile.TemporaryDirectory(prefix="diversity_combining-mbpp-") as tmp:
            work_dir = Path(tmp).resolve()
            script = work_dir / "candidate.py"
            started = work_dir / "interpreter-started"
            sentinel = work_dir / "assertion-passed"
            # The marker proves the interpreter ran, which is what separates a
            # wrong completion from a sandbox that never launched.
            source = (
                "from pathlib import Path as _HarnessPath\n"
                f"_HarnessPath({str(started)!r}).write_text('started', encoding='utf-8')\n"
                f"{imports}\n{code}\n{assertion}\n"
                "from pathlib import Path as _HarnessPath\n"
                f"_HarnessPath({str(sentinel)!r}).write_text('passed', encoding='utf-8')\n"
            )
            # Python compiles a whole file before running its first line, so a
            # completion that does not parse would never reach the marker and
            # its "F" would be misread as a sandbox that never started.
            # Compiling here, on the very interpreter the sandbox runs, settles
            # that case without changing how the candidate itself is executed.
            try:
                compile(source, str(script), "exec")
            except (SyntaxError, ValueError, RecursionError):
                outcomes.append("F")
                continue
            script.write_text(source, encoding="utf-8")
            try:
                result = subprocess.run(
                    _sandbox_command(script, work_dir),
                    cwd=work_dir,
                    env=deterministic_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=timeout_seconds,
                    check=False,
                    preexec_fn=lambda: _resource_limits(timeout_seconds),
                )
                passed = result.returncode == 0 and sentinel.is_file()
                if not passed and not started.is_file():
                    detail = (result.stderr or b"").decode("utf-8", "replace").strip()
                    raise RuntimeError(
                        "MBPP sandbox never started the interpreter for this "
                        "assertion, so an 'F' here would not mean the completion "
                        f"was wrong. Exit {result.returncode}; sandbox said: "
                        f"{detail or '<no stderr>'}"
                    )
                outcomes.append("P" if passed else "F")
            except subprocess.TimeoutExpired:
                if not started.is_file():
                    raise RuntimeError(
                        "MBPP sandbox timed out before the interpreter started, "
                        "so an 'F' here would not mean the completion was wrong."
                    ) from None
                outcomes.append("F")
    return "".join(outcomes)
