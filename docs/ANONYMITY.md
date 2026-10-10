# Before this goes public

The submission is double-blind, so the repository has to be anonymous until the
paper is not. Two separate things have to hold, and only one of them is about
file contents.

## 1. Nothing in the files names an author

`verify.sh` check 8 enforces this. It fails on any absolute path from a build
machine, on any home-directory path at all, and on the shape of a hosting URL.
A home directory in a path is enough to identify an author, and it survives
every other form of anonymisation.

It does **not** contain the author names. It used to, on the reasoning that
keeping them out of this note stopped the note becoming the leak it warns
about. That missed something: `verify.sh` ships. The list was published inside
the checker that was supposed to catch it, and a reviewer reading the file saw
two given names and the account the repository is hosted under.

The names now come from outside the tree:

```sh
SYN_NAMES='name1|name2' bash verify.sh
```

Run it that way before every push. A reviewer who does not set it still gets
every structural check; only someone who already knows the names can run the
one that looks for them, and by then they are not learning anything.

Five files carried them and were fixed: `tailcensor/cfg.py`,
`tailcensor/check_gt.py`, `tailcensor/smoke.py`, `phih/scripts/timing_all.py`
and `phih/scripts/run_effort.py`. All of them now take their roots from the
environment.

Add a name to `SYN_NAMES` when a new machine or account joins, rather than
relaxing the check. This file is itself scanned, and so is the checker, so
neither of them writes a name down.

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

## 2b. The embeddings are hosted somewhere else, and that host leaks differently

The 7.68 GiB of embeddings cannot go in git, so they sit on a file host and
`docs/DATA.md` points at it. That host is outside everything check 8 can see,
and consumer file hosts leak in a direction authors do not expect.

* **Towards the reviewer.** The owning account can appear in a file's details or
  sharing dialog. Upload from an account created for this, not from one that
  carries a real name, and do not reuse an account that appears anywhere else in
  the submission.
* **Towards the author.** This is the one that matters. A file host shows the
  owner who opened the file. A reviewer signed in to the host while downloading
  can end up in that log, which is a path to learning who reviewed the paper.
  Do not open the activity log, and prefer a host that does not keep one.

Neither risk is removed by the checksums or by anything in this repository. The
regeneration path in `docs/DATA.md` exists partly for this reason: a reviewer
who would rather not touch the host can rebuild the matrices from the corpora
and the adapter, both of which are here.

## Licence

MIT, in `LICENSE`. Two things it does not cover, and both are noted there:

* `native/third_party/hnswlib` is vendored unmodified under its own MIT licence.
* `patches/` holds diffs against other projects. Those projects are not
  redistributed here and stay under their own licences; applying a diff requires
  their source, which the reader clones at the commit the paper measured.
