# Commit Command

Create a focused git commit for the current change.

## Arguments

- No arguments are required.
- Optional text after `/commit` can provide commit intent or a preferred subject. If omitted, infer the subject from the staged diff and recent repo history.

## Commit Workflow

1. Inspect the working tree with `git status --short --branch`.
2. Review recent commit subjects before drafting a new one:
   ```bash
   git log --pretty=format:'%h %s' -10
   ```
3. Stage only the files that belong to the same concern.
4. Write the commit message using the conventions below.
5. Commit once the staged diff and subject describe one coherent change.
6. Do not push unless the user explicitly asks for a push.

## Commit Message Convention

Use the repository's current conventional-commit style:

```text
<type>: <imperative summary>
<type>(<scope>): <imperative summary>
```

Examples from recent history and repo-aligned docs:
- `docs: clarify commit message conventions`
- `fix(retrieval): render retrieved context as markdown`
- `fix(backend): resolve graph retrieval and verification failures`
- `refactor(tests): replace inline route stubs with real route factories`

## Rules

- Use a lowercase commit type: `fix`, `refactor`, `test`, `docs`, `chore`, or `feat` when it is truly a feature.
- Add a scope only when it materially narrows the subsystem, for example `backend`, `frontend`, `retrieval`, `tests`, `validation`, or `security`.
- Write the summary in imperative mood: `render`, `resolve`, `replace`, `add`.
- Keep the summary concise and specific. Describe the actual change or outcome, not generic activity.
- Prefer one concern per commit. If the change spans unrelated concerns, split it into separate commits.
- Use a body only when necessary to explain rationale, risk, or follow-up work.

## Avoid

Do not use vague or low-information subjects such as:
- `update stuff`
- `misc fixes`
- `address review comments`
- `wip`
- `changes`

Do not:
- mention internal mechanics when the user-visible or system-level effect is clearer
- combine unrelated fixes into one subject
- invent a scope when none helps the reader
- capitalize the type prefix or use sentence punctuation at the end

## Suggested Process For This Repo

- Start from the effect: what is fixed, refactored, or added?
- Compare with the last few commit subjects and match their level of specificity.
- If the diff is mostly tests, prefer `test(...)` or `fix(tests): ...` depending on whether behavior changed or only coverage changed.
- If the change corrects a bug in a specific subsystem, prefer `fix(<scope>): ...`.
- If the change is documentation-only, use `docs: ...`.

## Amending A Bad Subject

If the commit content is correct but the subject is not:

```bash
git commit --amend -m "<new subject>"
```

If that commit was already pushed, only rewrite the remote commit when you are explicitly correcting the just-pushed commit, and use:

```bash
git push --force-with-lease
```
