import argparse
import json
import os
import re
from typing import Dict, List, Optional


TIMESTAMP_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - (?P<msg>.*)$"
)
EPISODE_RE = re.compile(r"^-+ Episode (?P<episode>\d+) -+$")
SCORE_RE = re.compile(r"Your score \(min 0\.0, max 1\.0\):\s*(?P<score>[0-9.]+)")


def parse_log(log_path: str) -> List[Dict]:
    episodes: List[Dict] = []
    current_episode: Optional[Dict] = None
    current_turn: Optional[Dict] = None
    pending_generated_action: Optional[str] = None
    capture_observation = False

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            ts_match = TIMESTAMP_LINE_RE.match(line)

            if ts_match:
                msg = ts_match.group("msg")
                ep_match = EPISODE_RE.match(msg)

                if ep_match:
                    current_episode = {
                        "episode_id": int(ep_match.group("episode")),
                        "turns": [],
                    }
                    episodes.append(current_episode)
                    current_turn = None
                    pending_generated_action = None
                    capture_observation = False
                    continue

                capture_observation = False

                if current_episode is None:
                    continue

                if msg.startswith("Generated action:"):
                    pending_generated_action = msg.split("Generated action:", 1)[1].strip()
                    continue

                if msg.startswith("Action:"):
                    action_text = msg.split("Action:", 1)[1].strip()
                    current_turn = {
                        "generated_action": pending_generated_action or "",
                        "action": action_text,
                        "observation": "",
                    }
                    current_episode["turns"].append(current_turn)
                    pending_generated_action = None
                    continue

                continue

            if line.startswith("Observation:"):
                if current_episode is None:
                    continue

                if current_turn is None:
                    current_turn = {
                        "generated_action": pending_generated_action or "",
                        "action": "",
                        "observation": "",
                    }
                    current_episode["turns"].append(current_turn)
                    pending_generated_action = None

                observation_text = line.split("Observation:", 1)[1].lstrip()
                current_turn["observation"] = observation_text
                capture_observation = True
                continue

            if capture_observation and current_turn is not None:
                if current_turn["observation"]:
                    current_turn["observation"] += "\n" + line
                else:
                    current_turn["observation"] = line

    return episodes


def extract_score(episode: Dict) -> Optional[str]:
    for turn in episode["turns"]:
        obs = turn.get("observation", "")
        match = SCORE_RE.search(obs)
        if match:
            return match.group("score")
    return None


def write_scored_episodes(episodes: List[Dict], output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    written = 0

    for episode in episodes:
        score = extract_score(episode)
        if score is None:
            continue

        filename = f"episode_{episode['episode_id']}_score_{score}.jsonl"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            for turn in episode["turns"]:
                row = {
                    "generated_action": turn.get("generated_action", ""),
                    "action": turn.get("action", ""),
                    "observation": turn.get("observation", ""),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract episodes containing 'Your score' from a WebShop eval log and "
            "save each episode as one trajectory JSONL file."
        )
    )
    parser.add_argument("--log_path", required=True, help="Path to log file, e.g. cmd_results/test.log")
    parser.add_argument(
        "--output_dir",
        default="cmd_results/scored_trajectories",
        help="Directory to save extracted trajectory JSONL files",
    )
    args = parser.parse_args()

    episodes = parse_log(args.log_path)
    count = write_scored_episodes(episodes, args.output_dir)
    print(f"Parsed episodes: {len(episodes)}")
    print(f"Saved scored trajectories: {count}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
