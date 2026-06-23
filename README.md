# Sleepy Sentinel

This is a repo that trains a machine learning model to classify alertness levels.

This project requires `uv` to run.

## Train the model

Run `uv run train_alertness` to run training with the default parameters.

Here are some flags you might consider adjusting when running it though:

| Flag                       | Default | Type   |
| -------------------------- | ------- | ------ |
| --n-splits                 | 5       | int    |
| --validation-subject-count | 9       | int    |
| --epochs                   | 40      | int    |
| --batch-size               | 64      | int    |
| --learning-rate            | 1e-3    | float  |
| --random-seed              | 42      | int    |
| --wandb-project            | None    | string |

## Assignment 5

I ran training with `uv run train_alertness --epochs 100 --wandb-project sleepy-sentinel`.

The full writeup (that includes parts A & B) is at [docs/Assignment 5 Notes.md](./docs/Assignment%205%20Notes.md).

And although the `outputs` directory isn't tracked with Git, I included it in this zip since it includes the all the metrics and run data.
