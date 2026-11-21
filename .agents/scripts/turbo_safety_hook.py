"""PreToolUse safety hook for Antigravity Turbo mode.

Inspects `run_command` tool calls on stdin and forces manual confirmation
(`force_ask`) or denies (`deny`) destructive shell operations even when
Turbo permission mode is enabled.
"""

import json
import re
import sys

# Hard-deny patterns (never safe to auto-run without explicit user manual execution)
DENY_PATTERNS = [
    (r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh)\b", "Piping remote script directly into shell is blocked."),
    (r"\b(iex|invoke-expression)\b.*(irm|invoke-restmethod|iwr|invoke-webrequest)", "Remote PowerShell execution is blocked."),
    (r":\(\)\s*\{\s*:\|:\s*&\s*\};:", "Fork bomb pattern detected."),
]

# Force-ask patterns (overrides Turbo auto-approve and forces UI prompt)
FORCE_ASK_PATTERNS = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+|--recursive|--force)", "Recursive or forced deletion (`rm -r/-f`) requires confirmation."),
    (r"\bRemove-Item\b.*(-Recurse|-Force)", "PowerShell `Remove-Item -Recurse/-Force` requires confirmation."),
    (r"\b(del|erase|rmdir|rd)\b.*(/s|/q)", "Recursive Windows delete (`del/rmdir /s /q`) requires confirmation."),
    (r"\bgit\s+reset\s+--hard\b", "`git reset --hard` discards uncommitted changes."),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "`git clean -f` permanently deletes untracked files."),
    (r"\bgit\s+push\b.*(\s--force\b|\s-f\b|\s--force-with-lease\b)", "Force pushing to remote requires confirmation."),
    (r"\bgit\s+(checkout\s+--\s+\.|restore\s+\.)", "Discarding all local working tree changes requires confirmation."),
    (r"\bgit\s+branch\s+-D\b", "Force deleting a git branch requires confirmation."),
    (r"\b(format|mkfs|dd|diskpart)\b", "Disk/filesystem formatting command requires confirmation."),
]


def evaluate_command(cmd: str) -> dict:
    for pattern, reason in DENY_PATTERNS:
        if re.search(pattern, cmd, flags=re.IGNORECASE):
            return {"decision": "deny", "reason": f"[Turbo Safety Hook] {reason}"}

    for pattern, reason in FORCE_ASK_PATTERNS:
        if re.search(pattern, cmd, flags=re.IGNORECASE):
            return {"decision": "force_ask", "reason": f"[Turbo Safety Hook] {reason}"}

    return {"decision": "allow"}


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return

    tool_call = payload.get("toolCall", {})
    if tool_call.get("name") != "run_command":
        print(json.dumps({"decision": "allow"}))
        return

    cmd = tool_call.get("args", {}).get("CommandLine", "")
    result = evaluate_command(cmd)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
