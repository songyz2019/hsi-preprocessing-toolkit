## Setup commands

* Install dependencies: `uv sync`
* Start dev server: `uv run src/<project_name>/main.py`
* Run tests: `uv run test/<test_file>.py`

## Code style

* Follow [**The Zen of Python**](https://peps.python.org/pep-0020/), and [**The Unix Philosophy**](https://cscie2x.dce.harvard.edu/hw/ch01s06.html) (Eric Raymond Version)
* This is not and should not be a large project — avoid over-engineering.
* Use single quotes for symbolic strings, e.g. `'GREEN'`, `'about.title'`, `'key'`. Carefully change them.
  Use double quotes for long or user-facing strings, e.g. `"Please close this"`, `"Command not found"`.
* Avoid deeply nested `if` and `for` blocks. Early returns.
* Composition is better than inheritance: Prefer functional patterns. 
* Validate parameters outside core logic. If necessary, create a wrapper function (e.g. core `_compose` with a public `compose` wrapper).
* Prefer a **condensed structure across several files** instead of over-decoupling.
* We do the `Let it crash` and Gradio do the `Silent fallback`.
* Occam's Razor: Entities should not be multiplied unnecessarily. Minimize intermediate variables. Even if there're tons of parameters, do not make a dataclass.
* Don’t reinvent the wheel — and don’t buy juice when you already have a juicer. Smaller dependencies are better; more popular dependencies are better. No new dependencies are best.

## Extra

* Gradio can easily cause LLM hallucinations; be especially careful.
