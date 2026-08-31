#!/usr/bin/env bash
# Publish Vantage to a free Hugging Face Docker Space.
#
#   ./deploy/huggingface/push.sh <hf-username> [space-name]
#
# Hugging Face renders a Space from a README.md carrying a YAML frontmatter
# block, which GitHub would render as a stray table. So rather than keep that
# block in the repo, this script assembles a Space-shaped copy in a temporary
# directory: the frontmatter, then the real README, then the source tree.
#
# You are prompted for credentials by git itself. Use your HF username and an
# access token (https://huggingface.co/settings/tokens) with write permission as
# the password. Nothing is stored by this script.

set -euo pipefail

USER="${1:-}"
SPACE="${2:-vantage}"
if [ -z "$USER" ]; then
  echo "usage: $0 <hf-username> [space-name]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> staging a Space-shaped copy in $STAGE"
git -C "$REPO_ROOT" archive HEAD | tar -x -C "$STAGE"
cat "$REPO_ROOT/deploy/huggingface/space-header.md" > "$STAGE/README.md"
printf '\n' >> "$STAGE/README.md"
git -C "$REPO_ROOT" show HEAD:README.md >> "$STAGE/README.md"

echo "==> creating the Space repository"
echo "    If the Space does not exist yet, create it first at:"
echo "    https://huggingface.co/new-space   (SDK: Docker, hardware: CPU basic, free)"

cd "$STAGE"
git init -q -b main
git add -A
git -c user.email="noreply@huggingface.co" -c user.name="$USER" \
    commit -q -m "Deploy Vantage: self-correcting multi-agent data analyst"
git remote add space "https://huggingface.co/spaces/$USER/$SPACE"

echo "==> pushing to https://huggingface.co/spaces/$USER/$SPACE"
git push --force space main

echo
echo "Done. The Space builds the Dockerfile and will be live at:"
echo "  https://huggingface.co/spaces/$USER/$SPACE"
echo "  API docs:  https://$USER-$SPACE.hf.space/docs"
echo "  Health:    https://$USER-$SPACE.hf.space/health"
