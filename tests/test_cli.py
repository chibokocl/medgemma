"""Tests for medgemma.cli."""

from click.testing import CliRunner

from medgemma.cli import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "medgemma" in result.output


def test_info_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "Cache directory" in result.output
