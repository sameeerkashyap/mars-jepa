# Issue tracker: GitHub

Issues and specs live in `sameeerkashyap/mars-jepa` on GitHub. Use the `gh` CLI, prefixed with `rtk proxy`.

## Operations

- Verify the target before creating issues: `rtk proxy gh repo view --json nameWithOwner`. Use `--repo sameeerkashyap/mars-jepa` on issue commands.
- Create: `rtk proxy gh issue create --repo sameeerkashyap/mars-jepa --title "..." --body-file <path>`. Keep body files inside this project.
- Read: `rtk proxy gh issue view <number> --repo sameeerkashyap/mars-jepa --json number,title,body,labels,comments`.
- List: `rtk proxy gh issue list --repo sameeerkashyap/mars-jepa --state open --json number,title,body,labels`. Filter with `--label` when needed.
- Comment: `rtk proxy gh issue comment <number> --repo sameeerkashyap/mars-jepa --body-file <path>`.
- Label: `rtk proxy gh issue edit <number> --repo sameeerkashyap/mars-jepa --add-label "..."` or `--remove-label "..."`.
- Close: post the reason as a comment, then run `rtk proxy gh issue close <number> --repo sameeerkashyap/mars-jepa`.

Use native GitHub sub-issues and issue dependencies when supported. Otherwise, put `Part of #<parent>` and `Blocked by: #<number>` at the top of child issue bodies. An issue can start when all blockers are closed.

When a skill says to publish to the issue tracker, create a GitHub issue. When it says to fetch a ticket, read that issue.

File issues in this repository. Check the target explicitly in a fork.

## Pull requests as a triage surface

**PRs as a request surface: no.**
