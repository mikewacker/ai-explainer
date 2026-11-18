# Project Guidelines

- This project produces assets for an AI explainer: images, videos, and LaTeX equations.
- This project is a Python project that uses `uv`.

## Content

### Genre

- The genre for this AI explainer is a "Feynman explainer."
- This Richard Feynman quote defines the genre conventions: "I can’t understand anything in general
  unless I’m carrying along in my mind a specific example and watching it go."

### Examples

The AI explainer is built around three examples:

1. `number`
    - **Task:** \[regression\] guessing a number
    - **Data:** the number ( $y$ )
    - **Model:** $\hat{y} = w$
2. `wine`
    - **Task:** \[regression\] predicting a wine's proline, using malic acid and proanthocyanidins
    - **Data:** 3 samples from the [wine dataset](https://archive.ics.uci.edu/dataset/109/wine)
    - **Model:** $\hat{y} = w_1 x_1 + w_2 x_2$
3. `iris`
    - **Task:** \[binary classification\] predicting whether an iris flower is iris virginica or not 
        - An appendix explains the original multiclass classification task. 
    - **Data:** [iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
    - **Model:** neural network with 1 hidden layer of size 5

### Structure

The general template for an example is built around 4 sub-tasks:

1. **model:** building the model
2. **loss:** computing loss
3. **slope:** computing the gradient
4. **learn:** training the model (using gradient descent)

## Design

### Assets

- The assets for the AI explainer are stored in `golden/asset`.
    - A few assets are external; this project did not generate them.
- The top-level `ai_explainer` package has a subpackage for each example (e.g., `iris`).
- Each example has a top-level `__main__.py` module that produces the assets for that example.

### Tech Stack

`Dockerfile` and `pyproject.toml` define an environment that contains the tech stack:

- **PyTorch:** AI
- **Matplotlib (and `ffmpeg`):** figures (.png) and animations (.mp4)
- **Draw.io (and `pdftocairo`):** diagrams (.png)
- **Typst:** tables (.png)
- **LaTeX:** equations (.txt)

### API Design

Each example has a self-contained API (e.g., `import ai_explainer.wine as aix`):

- `world`: world or setting for the example
     - training data and test data
     - model class (i.e., `nn.Module` subclass) and train/test logic
     - tracing logic for "carrying along" a sample and "watching it go" through the model
- `canon`: canon for the example
     - reproducible canonical model, in both trained and untrained forms
     - canonical sample to carry along through the canonical model
- `asset`: tools for building hand-crafted assets
     - `asset` depends on `world`, but does not depend on `canon`.

`story` uses this API to assemble the assets; its role should be limited to high-level assembly.

## Testing

If you need to verify code changes, use the `$test` skill at `.agents/skills/test/SKILL.md`.

### Check

`make check` runs the following tasks:

- `uv run ruff check`
- `uv run ty check`
- `uv run ruff format --check`

### Strategy

- Tests for `story` only consist of "end-to-end" golden tests for the assets.
- Testing should be more concentrated for the components in `world` and `asset`.
- `asset` uses test data and test models so that changes to `canon` do not affect these tests.

### Environment

- `Dockerfile` defines the Docker image for the golden tests.
    - For CI, a GitHub Actions workflow sets up this environment. See: `.github/workflows/test.yml`
    - For local development, `docker compose run --rm ai-explainer` sets up a similar environment.
- Non-golden tests can run in any Linux/Mac environment.
