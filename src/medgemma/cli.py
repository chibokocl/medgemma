"""Click CLI for medgemma."""

import json
import sys

import click

from ._version import __version__


@click.group()
@click.version_option(__version__, prog_name="medgemma")
def cli():
    """MedGemma – Medical AI on Apple Silicon."""


@cli.command()
@click.argument("prompt")
@click.option("--image", default=None, help="Path to an image file.")
@click.option("--max-tokens", default=None, type=int, help="Max tokens to generate.")
@click.option("--temperature", default=None, type=float, help="Sampling temperature.")
@click.option("--model-path", default=None, help="Path to a local MLX model.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON with stats.")
@click.option("--no-stream", is_flag=True, help="Disable streaming output.")
def ask(prompt, image, max_tokens, temperature, model_path, as_json, no_stream):
    """Send a prompt to MedGemma."""
    from .client import MedGemma

    kwargs = {}
    if model_path:
        kwargs["model_path"] = model_path
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    mg = MedGemma(**kwargs)

    gen_kwargs = {}
    if image:
        gen_kwargs["image"] = image

    if no_stream or as_json:
        resp = mg.ask(prompt, **gen_kwargs)
        if as_json:
            click.echo(
                json.dumps(
                    {
                        "text": resp.text,
                        "completion_tokens": resp.completion_tokens,
                        "tokens_per_second": resp.tokens_per_second,
                        "elapsed_seconds": resp.elapsed_seconds,
                    },
                    indent=2,
                )
            )
        else:
            click.echo(resp.text)
    else:
        for chunk in mg.stream(prompt, **gen_kwargs):
            click.echo(chunk, nl=False)
        click.echo()


@cli.command()
@click.option("--local-path", default=None, help="Copy an existing local model.")
@click.option("--force", is_flag=True, help="Re-download / overwrite existing cache.")
def setup(local_path, force):
    """Download or set up the MedGemma model."""
    from .convert import setup_model

    click.echo("Setting up MedGemma model...")
    try:
        path = setup_model(force=force, local_path=local_path)
        click.echo(f"Model ready at: {path}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command()
def info():
    """Show model and cache information."""
    from .model import model_info

    mi = model_info()
    click.echo(f"Cache directory : {mi.cache_dir}")
    click.echo(f"Model in cache  : {'yes' if mi.cache_ready else 'no'}")
    click.echo(f"Model loaded    : {'yes' if mi.loaded else 'no'}")
    if mi.path:
        click.echo(f"Loaded from     : {mi.path}")
