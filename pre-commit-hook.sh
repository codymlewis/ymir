#!/bin/sh

# Get the updated python files
FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep '.py' | sed 's| |\\ |g')
[ -z "$FILES" ] && exit 0

# Sort the imports, and format all changed python files
echo "$FILES" | xargs isort -q
echo "$FILES" | xargs yapf -i -p -r

# Update docs if any python files in the ymir module were changed
# ROOT=$(git rev-parse --show-toplevel)
# echo "$FILES" | grep '^ymir' && pdoc --math -d restructuredtext -o "$ROOT/docs" "$ROOT/ymir" && git add "$ROOT/docs"

# Add the updated files
echo "$FILES" | xargs git add

exit 0