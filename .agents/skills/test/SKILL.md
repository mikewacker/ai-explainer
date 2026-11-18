---
name: test
description: Verify code changes.
---

# Goal

The high-level goal is to create a partial CI run that is fast:

- \[Correctness\] If the partial CI run passes, `make check` and `make test` should pass in a full CI run.
- \[Latency\] The partial CI run should be faster than a full CI run, which can be quite slow.
- \[Latency\] The partial CI run should fail fast.

# Assumptions

This skill assumes that the changes are in a testable state:

- If this assumption does not hold, you will need to adapt accordingly.
- Example: If a function has an incomplete implementation, `make check` or `make test` can fail.

# Check

Run `make check` first. If `make check` fails, stop and do not run tests.

# Test

## Groups

To fail fast, split the tests into 4 groups, and run (non-empty) groups in this order:

1. non-golden tests: `-m "not golden"`
   - Tests are located in a `world` package or an `asset` package.
2. golden "unit" tests: `-m "golden and not asset"`
   - Tests are located in an `asset` package or a `test_canon.py` module.
   - Golden files are stored in `golden/scrap` or `golden/model`.
3. golden "end-to-end" tests, excluding videos: `-m "asset and not video"`
   - Tests are located in a `story` package.
   - Golden files are stored in `golden/asset`.
4. golden "end-to-end" tests with videos: `-m "asset"`

Run all selected tests in each group. If a group fails, stop and do not run the next group.

## Limitations

For golden "end-to-end" test modules, you effectively have only 3 options for test selection:

- Run the entire test module.
- Run the entire test module, excluding videos.
- Skip the entire test module.

(This limitation is downstream of the module-level `produce()` fixture for these test modules.)

For golden "end-to-end" tests, this limitation also has implications for test grouping:

- If at least one video in a test module is affected, put the entire test module in group 4.
    - Corollary: Group 4 includes non-video tests, which is why it's `-m "asset"`, not `-m "video"`.
- If no videos in a test module are affected, put the entire test module in group 3.

## Selection

To run tests faster, isolate affected tests where feasible:

- For all groups, you can isolate affected examples (e.g., `tests/ai_explainer/number/`).
- If the changes only affect `story` packages and `golden/asset`, you can skip groups 1-2.
- Group 1: Since these tests are fast, don't get more granular than isolating affected examples.
- Group 2: Isolate affected tests where feasible.
- Groups 3-4: Isolate affected test modules where feasible.
    - Since end-to-end video tests are slow, only put a module in group 4 if a video is affected.

## Environment

- Group 1: Run non-golden tests in the local environment.
- Groups 2-4: Run golden tests in a Docker environment: `docker compose run --rm ai-explainer`

# Fix

## In General

- If the fix for a failed check or test is simple, then fix it.
    - `make fix` can fix some `make check` failures.
- Avoid large changes to fix failures.
- If you fix a failed check or test, follow these retry rules:
    - If you change the code, start over from `make check`.
    - If you only change golden files, retry affected tests in that group, and continue from there.

## Updating Goldens

- If changes to a golden file are expected, there are two ways to generate an updated file:
    - Run the binary in a Docker environment, e.g.: `uv run python -m ai_explainer.number loss`
    - If a golden test fails, the generated file is stored in `out/failures`.
- If changes to a golden file are not expected, you should generally not update that file.
- Do not visually inspect images or videos unless the user asks you to.

# Communication

- Do not communicate `make check` results to the user, unless there were unfixable failures.
- Provide the user a list of `pytest` commands that were run, and report unfixable test failures.
    - Only provide the `pytest` part; omit the `uv run` and `docker compose` parts.
- If applicable, provide the user a list of golden files that were affected (and/or renamed).
    - If the only change is the index in the filename, just state that some indexes were updated.
