#!/usr/bin/env bash
# Builds the folder to upload to Google Drive for the submission form (case section 09):
#   runs/submission/AuditGym/            the repository at HEAD (code, README, matcher and reward,
#                                        episodes, tests, policies, trajectories, design note, reports)
#   runs/submission/AuditGym.zip         the same as one archive
#   runs/submission/auditgym_demo.mp4    the demo video (scripts/make_demo_video.py)
# Only committed files go in: run it on a clean tree after the final commit.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is not clean: commit first" >&2
    exit 1
fi
video=runs/video/auditgym_demo.mp4
if [ ! -f "$video" ]; then
    echo "no $video: run 'uv run python scripts/make_demo_video.py' first" >&2
    exit 1
fi

out=runs/submission
rm -rf "$out"
mkdir -p "$out/AuditGym"
git archive --format=tar HEAD | tar -x -C "$out/AuditGym"
cp "$video" "$out/"
(cd "$out" && zip -qr AuditGym.zip AuditGym)

echo "commit $(git rev-parse --short HEAD)"
du -sh "$out/AuditGym" "$out/AuditGym.zip" "$out/auditgym_demo.mp4"
