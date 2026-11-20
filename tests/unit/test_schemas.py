"""Public files and truth of the fixture validate against the published JSON Schemas."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _registry() -> Registry:
    resources = []
    for path in SCHEMA_DIR.glob("*.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


REGISTRY = _registry()


def validator(name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, registry=REGISTRY)


@pytest.mark.parametrize("path", sorted(SCHEMA_DIR.glob("*.schema.json")), ids=lambda p: p.name)
def test_schemas_are_valid(path):
    Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_fixture_matches_schemas(tiny_episode):
    pub = tiny_episode.public
    validator("manifest").validate(pub.manifest)
    validator("policy").validate(pub.policy)
    for row in pub.ledger:
        validator("ledger_row").validate(row.to_dict())
    for inv in pub.invoices:
        validator("invoice").validate(inv.to_dict())
    for pay in pub.payments:
        validator("payment").validate(pay.to_dict())
    for fx in pub.fx:
        validator("fx_rate").validate(fx.to_dict())
    validator("truth").validate(tiny_episode.private.truth.to_dict())


def test_oracle_submission_matches_schema(tiny_episode):
    findings = [
        {
            "type": i.type,
            "affected_ids": list(i.affected_ids),
            "claimed_amount_minor": i.expected_amount_minor,
            "currency": i.currency,
            "evidence_ids": list(i.evidence_sets[0]),
            "explanation": "",
        }
        for i in tiny_episode.private.truth.issues
    ]
    validator("submission").validate({"episode_id": "ep_TINY000001", "findings": findings})
    validator("action").validate(
        {"action": "submit", "args": {"findings": {"episode_id": "x", "findings": findings}}}
    )
