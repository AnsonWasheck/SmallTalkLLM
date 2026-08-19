# Publishing the GitHub repository

## One-time setup

Create an empty GitHub repository named `SmallTalkLLM` without adding a README,
license, or gitignore (they already exist locally), then set the remote:

```bash
git remote add origin https://github.com/AnsonWasheck/SmallTalkLLM.git
git push -u origin main
git push -u origin develop
git push origin v0.1.0
```

If you prefer SSH:

```bash
git remote set-url origin git@github.com:AnsonWasheck/SmallTalkLLM.git
```

## GitHub settings to enable

1. Set `main` as the default branch.
2. Protect `main`: pull request required, CI required, no force-pushes, no branch deletion.
3. Allow `develop` to receive integration pull requests.
4. Enable Dependabot security updates.
5. Add a repository description and topics: `small-language-model`, `conversation`,
   `language-model`, `machine-learning`, `research`, `pytorch`.
6. Publish the first release from tag `v0.1.0`.
7. Add checkpoint and dataset links only after their cards, licenses, and manifests
   are complete. Do not upload the local `artifacts/`, `models/`, or `data/raw/` trees.

## Release convention

Source releases use `vMAJOR.MINOR.PATCH`. A model release should additionally
identify the student config, tokenizer checksum, dataset manifest, training
commit, and evaluation tag, for example:

```text
smalltalk-7m-r1
student: smalltalk-7m (6,689,024 params)
tokenizer: tokenizer-4096/<sha256>
data: teacher-v2/<manifest-sha256>
code: v0.2.0
eval: pre_teacher_v2
```
