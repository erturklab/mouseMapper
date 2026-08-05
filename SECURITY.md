# Security Policy

Thank you for helping keep **mouseMapper** and its users safe.

This document explains how to report security issues in a way that protects
both the project and the people who depend on it.

## Supported versions

mouseMapper is an active research codebase. Security updates are applied to
the latest commit on the `main` branch. Older commits, tags, or forks are not
actively patched.

| Version       | Supported          |
| ------------- | ------------------ |
| `main` branch | :white_check_mark: |
| Older commits | :x:                |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues
or pull requests.** Public reports give attackers a head start before a fix
is available.

Instead, please use one of the following private channels:

1. **GitHub private vulnerability reporting** (preferred)
   - Go to the [Security tab](../../security) of this repository
   - Click **Report a vulnerability**
   - Fill in the form with as much detail as you can

2. **Email the maintainers**
   - Contact the corresponding author listed in [`CITATION.cff`](./CITATION.cff)
     or on the project's publication page
   - Subject line: `mouseMapper security report`

### What to include

To help us triage quickly, please include where possible:

- A clear description of the issue and its potential impact
- Steps to reproduce, or a minimal proof-of-concept
- The affected file(s), module(s), or commit hash
- Your environment (OS, Python version, relevant package versions)
- Any suggested mitigations or patches

### What to expect

- We aim to acknowledge new reports within a reasonable time frame.
- Once a report is validated, we will work on a fix and coordinate disclosure
  with the reporter.
- With your permission, we are happy to credit you in the release notes or
  commit message once the fix is public.

## Scope

This policy covers the code and configuration in this repository. Issues in
third-party dependencies (for example, `nnU-Net`, `Voreen`, `VTK`, or PyPI
packages listed in the various `requirements.txt` files) should be reported
upstream to the respective projects. We are still happy to hear about them so
we can update pinned versions or document workarounds.

## Out of scope

The following are generally **not** considered security vulnerabilities in
mouseMapper itself:

- Bugs that require attacker-controlled training data or model weights with
  no path to remote code execution
- Issues only reproducible on unsupported, heavily modified, or end-of-life
  operating systems
- Denial-of-service caused solely by passing very large inputs (this is a
  research pipeline, not a hardened service)
- Findings from automated scanners without a demonstrated impact

Thank you for taking the time to make mouseMapper safer for the community.
