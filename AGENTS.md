# Rules for the coding agent

1. Read `README.md`. It is the spec.
2. Edit **only the files the run's prompt lists**. Everything else is read-only, including
   `README.md`, `AGENTS.md` and `tests/`. Never change a signature marked `# INTERFACE`.
3. Done means `pytest` passes and the outputs the prompt asks for exist in the exact format given.
   Never weaken, skip or `xfail` a test; if you think one is wrong, say so in your final message.
4. Need an interface change or another file? Stop and say so in your final message. Do not work around it.
5. No git commands. No installs.
6. Python: always `conda run -p "C:\Users\leolu\miniconda3\envs\research" python ...` from the repo
   root. Do not run `pip install -e .`.
7. Code: type hints, NumPy docstrings with shapes and units, float64, explicit
   `numpy.random.Generator` seeds, no global state, sparse T only.
