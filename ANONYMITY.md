# Before this goes public

The submission is double-blind, so the repository has to be anonymous until the
paper is not. Two separate things have to hold, and only one of them is about
file contents.

## 1. Nothing in the files names an author

`verify.sh` check 8 enforces this. It fails on any absolute path from a build
machine, on any home-directory path at all, and on the account names that have
appeared in these trees. Those names are listed in the check itself, not here,
so that this note does not become the leak it is warning about. A home directory
in a path is enough to identify an author, and it survives every other form of
anonymisation.

Four files carried them and were fixed: `tailcensor/cfg.py`,
`tailcensor/check_gt.py`, `tailcensor/smoke.py`, `phih/scripts/timing_all.py`
and `phih/scripts/run_effort.py`. All of them now take their roots from the
environment.

Extend the name list in check 8 when a new machine joins, rather than relaxing
it. Note that this file is itself scanned, so write about the names rather than
writing them down.

## 2. The hosting does not name an author

Check 8 cannot see this part. A repository under a personal account is
identifying whatever the files say, and so is its commit history: author names
and email addresses live in the commit metadata, not the tree.

So do not push this to a personal account before the decision. Use an anonymised
mirror -- anonymous.4open.science, or an anonymous archive on a neutral host --
and keep the real repository private until acceptance. If you must create the
real repository now, create it private and squash the history before it is ever
made public, because `git log` will otherwise carry every author line.

## 3. What about being scooped

The honest answer is that the risk is low and it is not the reason to keep this
closed. The contribution is a training procedure plus a measurement campaign;
reproducing it takes a GPU, a week of machine time and the same taste in
controls. Publishing early would not meaningfully raise the risk.

The reason to keep it closed is that publishing it under a personal account
breaks anonymity, and that is not recoverable once an index has crawled it.

After acceptance, make it public and keep the anonymised mirror alive: reviewers
and readers cite the link they were given.

## Licence

MIT, in `LICENSE`. Two things it does not cover, and both are noted there:

* `lab/cpp/third_party/hnswlib` is vendored unmodified under its own MIT licence.
* `patches/` holds diffs against other projects. Those projects are not
  redistributed here and stay under their own licences; applying a diff requires
  their source, which the reader clones at the commit the paper measured.
