#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision=c7622b41d1eee5f18082dab8beed2ee2db5b7cde
origin=https://github.com/propeller-heads/fynd
patch="$root/patches/fynd-initializer.patch"
mkdir -p "$root/work"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

prepare() {
    local directory="$1"
    if [[ ! -e "$directory" ]]; then
        git init --quiet "$directory"
        git -C "$directory" remote add origin "$origin"
        git -C "$directory" fetch --depth 1 origin "$revision"
        git -C "$directory" checkout --quiet --detach FETCH_HEAD
    fi
    [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || {
        echo "Wrong Fynd revision in $directory; refusing to replace it." >&2
        exit 1
    }
    local checkout_origin
    checkout_origin="$(git -C "$directory" remote get-url origin)"
    # Existing patched checkouts may have been cloned from the verified clean checkout.
    [[ "$checkout_origin" == "$origin" ||
       ( "$directory" == "$root/work/fynd-patched" &&
         "$checkout_origin" == "$root/work/fynd-clean" ) ]] || {
        echo "Unexpected Fynd origin in $directory." >&2
        exit 1
    }
}

matches_index() {
    local directory="$1" index="$2"
    GIT_INDEX_FILE="$index" git -C "$directory" diff --no-ext-diff --no-textconv --quiet -- &&
        [[ -z "$(GIT_INDEX_FILE="$index" git -C "$directory" ls-files --others --exclude-standard)" ]]
}

prepare "$root/work/fynd-clean"
GIT_INDEX_FILE="$scratch/clean-index" git -C "$root/work/fynd-clean" read-tree "$revision"
matches_index "$root/work/fynd-clean" "$scratch/clean-index" || {
    echo "Clean Fynd checkout has local changes; refusing to replace them." >&2
    exit 1
}
prepare "$root/work/fynd-patched"
GIT_INDEX_FILE="$scratch/patched-index" git -C "$root/work/fynd-patched" read-tree "$revision"
if matches_index "$root/work/fynd-patched" "$scratch/patched-index"; then
    git -C "$root/work/fynd-patched" apply --check "$patch"
    git -C "$root/work/fynd-patched" apply "$patch"
fi
# Build the exact expected tree in a temporary index. This checks every upstream and patched
# file without changing the checkout's index or depending on diff hash abbreviations.
GIT_INDEX_FILE="$scratch/patched-index" git -C "$root/work/fynd-patched" apply --cached "$patch"
matches_index "$root/work/fynd-patched" "$scratch/patched-index" || {
    echo "Patched Fynd differs from the published patch; refusing to replace changes." >&2
    exit 1
}
printf 'Verified Fynd %s, clean and patched.\n' "$revision"
