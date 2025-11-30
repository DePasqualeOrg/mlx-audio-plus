# Filtering Git History to Only Current Files

This approach removes all files from history that don't exist in the latest commit. Useful for cleaning up contributor stats when large files were added and later deleted.

## Steps

### 1. Generate list of current files

```bash
git ls-files > /tmp/files-to-keep.txt
```

### 2. Filter history

```bash
git filter-repo --paths-from-file /tmp/files-to-keep.txt --force
```

## Result

- Files that were added then deleted → erased from history
- Contributor stats only reflect files that remain today
- No need to hunt down individual offending files

## Caveat

If a file was **renamed**, history from before the rename may be lost. To check for this:

```bash
git filter-repo --paths-from-file /tmp/files-to-keep.txt --force --analyze
cat .git/filter-repo/analysis/renames.txt
```
