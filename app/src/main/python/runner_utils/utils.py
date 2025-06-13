import os
import re
import socket
import subprocess
import requests
from pathlib import Path
from typing import Optional


def run_command(cmd: list[str], cwd: Optional[Path] = None):
    print(f"🔧 Running: {' '.join(cmd)} (in {cwd or Path.cwd()})")
    subprocess.run(cmd, check=True, cwd=cwd)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def comment_on_issue(repo: str, issue_number: int, comment: str, token: str):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    print(f"💬 Commenting on issue #{issue_number}: {comment[:60]}...")
    response = requests.post(url, headers=headers, json={"body": comment})
    response.raise_for_status()


def close_issue(repo: str, issue_number: int, token: str):
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    print(f"✅ Closing issue #{issue_number}")
    response = requests.patch(url, headers=headers, json={"state": "closed"})
    response.raise_for_status()


def parse_yaml_from_issue_body(body: str) -> Optional[dict]:
    try:
        match = re.search(r"```yaml\s+(.*?)```", body, re.DOTALL)
        if not match:
            raise ValueError("No valid YAML block found")
        yaml_str = match.group(1).strip()

        import yaml
        return yaml.safe_load(yaml_str)
    except Exception as e:
        print("YAML parsing error:", e)
        return None
