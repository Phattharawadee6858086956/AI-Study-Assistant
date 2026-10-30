# Repository Guidelines

## Project Structure

This repository is currently an empty project workspace. No source, tests, assets, or build configuration are present yet. As the project grows, keep application code in clearly named source directories (such as `src/`), tests in `tests/` or alongside the code they cover, and static files in `assets/` or `public/`. Update this guide when the actual layout is established.

## Build, Test, and Development

There are no build, test, or local-run commands configured at present. Do not assume a particular language or package manager. When adding tooling, document the canonical setup and commands here (for example, dependency installation, running locally, and the full test suite), and ensure they work from the repository root.

## Coding Style

No formatter, linter, or language-specific style rules are configured. Follow the conventions of the language and framework introduced by each change. Use descriptive, consistent file and identifier names, keep modules focused, and apply the project's formatter or linter once one is configured. Avoid adding generated files or dependencies without a clear project need.

## Testing

No test framework or coverage requirement is established. Add tests with new behavior and regression fixes, using names that describe the behavior under test (for example, `test_<behavior>` or `<feature>.test.<ext>`, matching the chosen framework). Record the command contributors should run and any required environment setup when test tooling is introduced.

## Commits and Pull Requests

There is no Git history available in this workspace to infer commit conventions. Use short, imperative commit subjects that describe one change (for example, `Add input validation`). Pull requests should explain the purpose and notable implementation choices, link any related issue, and include test results. Add screenshots only when a change affects a visual interface.

## Configuration and Secrets

Keep credentials and machine-specific settings out of version control. If configuration is needed, provide safe example values in a clearly named sample file and document how developers supply local secrets.
