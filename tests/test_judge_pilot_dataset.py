import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vlm_anchor.judge_pilot_data import (
    PilotSample,
    iter_pilot_arms,
    write_manifest_jsonl,
    load_manifest_jsonl,
)


class TestIterPilotArms(unittest.TestCase):
    def test_iter_arms_yields_three_records_per_sample(self) -> None:
        s = PilotSample(
            sample_id="LRVInstruction-000000008757",
            prompt="Are there any stop signs with yellow writing on them?",
            image_path=Path("/x/img.png"),
            chosen_response="No, the stop sign in the image does not have yellow writing.",
            anchor_image_path=Path("/x/anchor.png"),
            anchor_image_masked_path=Path("/x/anchor_masked.png"),
        )
        arms = list(iter_pilot_arms(s))
        self.assertEqual([a["arm"] for a in arms], ["b", "a", "m"])
        self.assertEqual(len(arms[0]["images"]), 1)
        self.assertEqual(arms[0]["images"][0], s.image_path)
        self.assertEqual(arms[1]["images"], [s.image_path, s.anchor_image_path])
        self.assertEqual(arms[2]["images"], [s.image_path, s.anchor_image_masked_path])


class TestManifestRoundTrip(unittest.TestCase):
    def test_write_and_load_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            samples = [
                PilotSample(
                    sample_id=f"id-{i}",
                    prompt=f"prompt-{i}",
                    image_path=Path(f"/x/{i}.png"),
                    chosen_response=f"response-{i}",
                    anchor_image_path=Path("/x/anchor.png"),
                    anchor_image_masked_path=Path("/x/anchor_masked.png"),
                )
                for i in range(3)
            ]
            write_manifest_jsonl(samples, path)
            loaded = load_manifest_jsonl(path)
            self.assertEqual(len(loaded), 3)
            for orig, back in zip(samples, loaded):
                self.assertEqual(orig.sample_id, back.sample_id)
                self.assertEqual(orig.prompt, back.prompt)
                self.assertEqual(orig.image_path, back.image_path)
                self.assertEqual(orig.chosen_response, back.chosen_response)


if __name__ == "__main__":
    unittest.main()
