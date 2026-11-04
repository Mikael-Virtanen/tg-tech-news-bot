"""Static checks of the workflow files: the failures worth catching before GitHub runs them."""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in (ROOT / ".github" / "workflows").glob("*.yml")}
SECRET_NAMES = ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "TELEGRAPH_ACCESS_TOKEN")


def triggers(doc):
    return doc.get("on", doc.get(True))  # YAML 1.1 parses a bare `on:` key as boolean True


def only_job(doc):
    (job,) = doc["jobs"].values()
    return job


def test_both_workflows_exist():
    assert set(WORKFLOWS) == {"bot.yml", "digest.yml"}


@pytest.mark.parametrize("name", WORKFLOWS)
def test_schedules_are_valid_and_avoid_the_congested_top_of_the_hour(name):
    crons = [entry["cron"] for entry in triggers(WORKFLOWS[name])["schedule"]]
    assert crons
    for cron in crons:
        minute, hour, dom, month, dow = cron.split()  # exactly five fields
        assert minute.isdigit() and minute != "0", f"{cron}: avoid :00, the most delayed minute"
        assert not minute.startswith("*/"), f"{cron}: GitHub's minimum interval is 5 minutes"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_state_is_protected_by_a_shared_non_cancelling_concurrency_group(name):
    concurrency = WORKFLOWS[name]["concurrency"]
    assert concurrency["group"] == "news-bot-state" and concurrency["cancel-in-progress"] is False
    assert WORKFLOWS[name]["permissions"] == {"contents": "write"}  # nothing broader than needed


@pytest.mark.parametrize("name", WORKFLOWS)
def test_manual_runs_default_to_a_safe_dry_run(name):
    dispatch = triggers(WORKFLOWS[name])["workflow_dispatch"]["inputs"]["dry_run"]
    assert dispatch["type"] == "boolean" and dispatch["default"] is True


@pytest.mark.parametrize("name", WORKFLOWS)
def test_secrets_are_only_exposed_to_the_step_that_needs_them(name):
    job = only_job(WORKFLOWS[name])
    assert not any(secret in job["env"] for secret in SECRET_NAMES)  # not job-wide
    bot_steps = [s for s in job["steps"] if "src.main" in s.get("run", "")]
    assert len(bot_steps) == 1 and any(secret in bot_steps[0]["env"] for secret in SECRET_NAMES)
    for step in job["steps"]:
        if step is not bot_steps[0]:
            assert not any(secret in step.get("env", {}) for secret in SECRET_NAMES)


@pytest.mark.parametrize("name", WORKFLOWS)
def test_state_is_restored_before_and_saved_after_with_the_right_guards(name):
    steps = only_job(WORKFLOWS[name])["steps"]
    runs = [s.get("run", "") for s in steps]
    restore = next(i for i, r in enumerate(runs) if "state_pull.sh" in r)
    bot = next(i for i, r in enumerate(runs) if "src.main" in r)
    save = next(i for i, r in enumerate(runs) if "state_push.sh" in r)
    assert restore < bot < save
    condition = steps[save]["if"]
    assert "always()" in condition                                  # saved even if the bot crashed or timed out
    assert "steps.restore.outcome == 'success'" in condition        # never push if the restore did not work
    assert "DRY_RUN != 'true'" in condition                         # dry runs leave no trace
    assert steps[restore]["id"] == "restore"
    assert steps[bot]["timeout-minutes"] < only_job(WORKFLOWS[name])["timeout-minutes"]  # leaves time for the save step


@pytest.mark.parametrize("name", WORKFLOWS)
def test_actions_are_pinned_to_a_major_version_and_python_is_modern(name):
    steps = only_job(WORKFLOWS[name])["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"actions/[\w-]+@v\d+", step["uses"]), step["uses"]
    setup = next(s for s in steps if s.get("uses", "").startswith("actions/setup-python"))
    assert setup["with"]["python-version"] == "3.12"


def test_commands_match_what_the_cli_accepts():
    from src.main import main

    assert only_job(WORKFLOWS["bot.yml"])["steps"][-3]["run"].strip() == "python -m src.main run"
    assert only_job(WORKFLOWS["digest.yml"])["steps"][-2]["run"].strip() == "python -m src.main digest"
    for command in ("run", "digest"):
        with pytest.raises(SystemExit) as info:
            main([command, "--help"])
        assert info.value.code == 0


def test_digest_job_is_inert_unless_digest_mode_is_selected():
    condition = only_job(WORKFLOWS["digest.yml"])["if"]
    assert "vars.PUBLISH_MODE == 'digest'" in condition and "workflow_dispatch" in condition
    assert "if" not in only_job(WORKFLOWS["bot.yml"])


def test_every_variable_the_workflows_pass_is_one_the_code_reads():
    """Catches typos like ITEM_PER_SOURCE: the workflow would pass a variable nothing ever reads."""
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    for name, doc in WORKFLOWS.items():
        for var in only_job(doc)["env"]:
            if var not in ("GITHUB_TOKEN",):
                assert var in source, f"{name} passes {var} but no module reads it"


@pytest.mark.parametrize("name", WORKFLOWS)
def test_single_mode_is_the_pinned_default_everywhere(name):
    """Unset repository variable => single mode, decided by the workflow itself, not by luck."""
    from src.config import Config

    expression = only_job(WORKFLOWS[name])["env"]["PUBLISH_MODE"]
    assert "vars.PUBLISH_MODE || 'single'" in expression
    assert Config.from_env({}).publish_mode == "single"
    assert Config.from_env({"PUBLISH_MODE": ""}).publish_mode == "single"  # what GitHub passes for an unset variable
    assert re.search(r"^PUBLISH_MODE=single\b", (ROOT / ".env.example").read_text(), re.MULTILINE)


def test_digest_workflow_does_nothing_in_single_mode():
    """With PUBLISH_MODE unset or 'single', the scheduled digest job is skipped (no digest, no state write)."""
    condition = only_job(WORKFLOWS["digest.yml"])["if"]
    assert "vars.PUBLISH_MODE == 'digest'" in condition       # only the *explicit* variable can enable it
    assert "vars.PUBLISH_MODE || 'single'" not in condition   # the default must not enable it
