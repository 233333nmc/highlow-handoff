"""Tests for the lightweight cross-service contracts."""

from __future__ import annotations

import unittest

from highlow.contracts import (
    ActionEnvelope,
    EpisodeContext,
    Modality,
    ObservationEnvelope,
    RequestContext,
)


class ContractTest(unittest.TestCase):
    def test_contracts_do_not_require_model_dependencies(self) -> None:
        episode = EpisodeContext(task_id="pick", instruction="pick the object")
        observation = ObservationEnvelope(
            episode_id=episode.episode_id,
            step_id=0,
            timestamp_s=0.0,
            modalities={
                "robot.state": Modality(
                    name="robot.state",
                    schema_id="robot.state@1",
                    data=[0.0, 1.0],
                )
            },
        )
        action = ActionEnvelope(
            episode_id=episode.episode_id,
            step_id=0,
            schema_id="robot.action@1",
            data=[0.5],
        )

        self.assertEqual("robot.state@1", observation.require("robot.state").schema_id)
        self.assertEqual("robot.action@1", action.schema_id)

    def test_contracts_copy_mutable_metadata(self) -> None:
        metadata = {"worker": "a"}
        request = RequestContext(metadata=metadata)
        metadata["worker"] = "b"

        self.assertEqual("a", request.metadata["worker"])
        with self.assertRaises(TypeError):
            request.metadata["new"] = "value"  # type: ignore[index]

    def test_modality_key_must_match_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            ObservationEnvelope(
                episode_id="episode",
                step_id=0,
                timestamp_s=0.0,
                modalities={
                    "camera": Modality(
                        name="different",
                        schema_id="image.rgb@1",
                        data=b"",
                    )
                },
            )


if __name__ == "__main__":
    unittest.main()
