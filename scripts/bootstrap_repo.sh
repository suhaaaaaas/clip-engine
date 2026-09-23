#!/usr/bin/env bash
# Create the GitHub repo and push this skeleton. Run once, from the repo root.
#
#   ./scripts/bootstrap_repo.sh [repo-name] [public|private]
#
# Requires the GitHub CLI: https://cli.github.com  (brew install gh)

set -euo pipefail

REPO_NAME="${1:-clip-engine}"
VISIBILITY="${2:-private}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh (GitHub CLI) is not installed."
  echo "  macOS:  brew install gh"
  echo "  Linux:  https://github.com/cli/cli/blob/trunk/docs/install_linux.md"
  exit 1
fi

# Triggers the browser login flow if you are not already authenticated.
if ! gh auth status >/dev/null 2>&1; then
  echo "==> Logging in to GitHub"
  gh auth login
fi

GH_USER="$(gh api user --jq .login)"
echo "==> Authenticated as ${GH_USER}"

if [ ! -d .git ]; then
  git init -b main
fi

git add -A
if git diff --cached --quiet; then
  echo "==> Nothing new to commit"
else
  git commit -m "Scaffold clip-engine: stage modules, eval harness, golden set format"
fi

if gh repo view "${GH_USER}/${REPO_NAME}" >/dev/null 2>&1; then
  echo "==> Repo ${GH_USER}/${REPO_NAME} already exists; pushing to it"
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "https://github.com/${GH_USER}/${REPO_NAME}.git"
  git push -u origin main
else
  echo "==> Creating ${VISIBILITY} repo ${GH_USER}/${REPO_NAME}"
  gh repo create "${REPO_NAME}" \
    --"${VISIBILITY}" \
    --source=. \
    --remote=origin \
    --description "Turn long video podcasts into short vertical clips" \
    --push
fi

echo
echo "==> Done: https://github.com/${GH_USER}/${REPO_NAME}"
echo "    Add your partner:  gh repo edit ${GH_USER}/${REPO_NAME} --add-collaborator <username>"
