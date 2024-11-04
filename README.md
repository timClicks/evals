# Stencila Evals

Some text here.

## Development

For development, you’ll need to install the following dependencies:

* [uv][uv]
* [just][just]

The following will get your started:

```sh
just init
```

Once `uv` is installed, you can use it to install some additional tools:

```sh
uv tool install ruff
uv tool install pyright
```

The `just` tool is used to run development-related tasks.
The `check` command runs all linting and tests, for example:

```sh
just check
```

The `justfile` has some common commands that you might want to run.
To run anything within the virtual environment, you need to use `uv run <command>`.
Alternatively, you can install [direnv],
and have the virtual environment activated automatically.
Look here for more details about [direnv and uv][uv-direnv].

[uv-direnv]: https://github.com/direnv/direnv/wiki/Python#uv
[direnv]: https://direnv.net/
[uv]: https://docs.astral.sh/uv/
[just]: https://github.com/casey/just
