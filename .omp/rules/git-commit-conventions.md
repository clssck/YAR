---
description: Match repository commit conventions when running git commit directly
condition: "git\s+commit\b"
scope: "tool:bash"
---

**When running `git commit` directly instead of using `/commit`, match this repository's commit conventions.**

## Required workflow

1. Check recent subjects first:
   ```bash
   git log --pretty=format:'%h %s' -10
   ```
2. Stage only one coherent concern.
3. Write the subject in the repository's conventional format.

## Required format

```text
<type>: <imperative summary>
<type>(<scope>): <imperative summary>
```

Examples:
- `docs: clarify commit message conventions`
- `fix(retrieval): render retrieved context as markdown`
- `fix(backend): resolve graph retrieval and verification failures`
- `refactor(tests): replace inline route stubs with real route factories`

## Rules

- Use a lowercase type such as `fix`, `refactor`, `test`, `docs`, `chore`, or `feat`.
- Add a scope only when it helps narrow the subsystem.
- Keep the summary concise, specific, and imperative.
- Describe the actual change or effect, not generic activity.
- Use a body only when rationale, risk, or follow-up needs explanation.

## Avoid

Do not use subjects like:
- `update stuff`
- `misc fixes`
- `address review comments`
- `wip`
- `changes`

Do not combine unrelated concerns into one commit subject.
