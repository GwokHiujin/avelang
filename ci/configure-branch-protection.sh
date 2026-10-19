#!/usr/bin/env bash
set -euo pipefail

repository="${1:-GwokHiujin/avelang}"
branch="${2:-master}"
required_check="Build and test on gfx950"

[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || {
  echo "invalid repository: $repository" >&2
  exit 2
}
[[ "$branch" =~ ^[A-Za-z0-9._/-]+$ ]] || {
  echo "invalid branch: $branch" >&2
  exit 2
}

command -v gh >/dev/null 2>&1 || {
  echo "gh is required" >&2
  exit 1
}
gh auth status >/dev/null

payload="$(mktemp)"
trap 'rm -f "$payload"' EXIT

cat >"$payload" <<JSON
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["${required_check}"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  "repos/${repository}/branches/${branch}/protection" \
  --input "$payload"

gh api \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2026-03-10" \
  "repos/${repository}/branches/${branch}/protection" \
  --jq '{strict: .required_status_checks.strict, checks: .required_status_checks.contexts, enforce_admins: .enforce_admins.enabled, pull_requests_required: (.required_pull_request_reviews != null), force_pushes: .allow_force_pushes.enabled, deletions: .allow_deletions.enabled}'
