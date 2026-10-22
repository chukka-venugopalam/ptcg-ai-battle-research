# Research synthesis (second pass) — what public evidence says about ladder rating

Sources read: Fusic/Zenn writeup (final 797.4, rank 1160/6807), Jun-Morita repo (LB 911, silver cut ~933),
wmh/ptcg-abc README, oscar-chw (709.1), hotorch (742.8), Kaggle competition pages. Kaggle itself was not reachable
from the sandbox, so Kaggle discussion/writeup posts (Team Magist, Solution Overview, "Last week rush", Final Meta Report)
are known only through the Fusic article's description of them. Treat those secondhand claims as UNVERIFIED.

## Evidence that changes the plan
1. **Local win-rate gains did not move the ladder.** Fusic measured +1.6 pt local gains (tight SE) for an agent whose six
   confirmed ladder scores were 727.8-818.8; the *same file* resubmitted scored 898.2 and 835.8; a hand-tuned Alakazam family
   spanned 862.7-983.8 after a first 993.8. Ladder noise per submission is roughly +/-50. A single score cannot validate a change.
2. **Deck/archetype family dominates the ceiling.** Grimmsnarl behavior-cloned agents peaked ~826 (Fusic) and ~781 here (B-v1);
   public Alakazam-family agents reached 861-994; Jun-Morita saw a transient 1123 with an anti-Crustle specialist.
   wmh found a simple deck piloted cleanly beat "stronger" combo decks on the ladder.
3. **Local rule-agent opponents overstate strength.** B-v1 beats the sample rule agents 58-86% locally yet sits at ~781 on the ladder:
   the ladder contains much stronger (BC+RL+search) Dragapult/Ogerpon pilots. Our "Dragapult 0.78" is vs Scio's rule:dragapult, not those.
4. **Ladder meta (Fusic's 4,501-game count, Aug 17):** Dragapult 38.2%, Slowking/Academy at Night 20.8%, Ogerpon 18.3%, Alakazam 6.9%,
   Crustle 4.9%, Froslass/Lopunny 3.6%, Rocket 1.9%, Lucario 1.8%, Kangaskhan 1.6%, Festival 0.7%.
   **Slowking (20.8%) has no opponent in our benchmark: UNKNOWN performance.** Our grid covers ~70% of this field.
5. **Going first is worth ~+8 pt (Fusic).** Verified here: B-v1 already answers IS_FIRST with option 0 whenever asked
   (100/100), and the chooser always goes first. No free gain.
6. **Top solutions used far more compute than available here** (reported 1.67e7-param model/1.35e8 steps; 1.17e8-param/1.68e9 steps).
   A credible 1200+ path needs Kaggle-scale GPU time, not a sandbox.
7. **Data filtering:** restricting BC teachers to >=1075-rated demonstrators raised held-out top-k (Jun-Morita: 0.8024 vs 0.7836); >=1150 hurt.
   Top-k agreement is a weak proxy: Fusic's ~80% agreeing Alakazam BC lost 8.3 pt.

## Measured here (all seat-alternating; Wilson CI)
* B-v1 over 13 cells x 200 games: overall 0.686; weak: Abomasnow 0.405, v10 0.585, Ogerpon-deck(B-v1 piloted) ~0.48 pooled.
* Meta-weighted local score over the ~70% of the field we can represent: ~0.70 — an upper-bound-style number, not a rating.
* **Policy-vs-policy cells vary more than their CI.** Ogerpon cell: 0.53 (n=200), 0.485 (n=200), 0.49 (n=100), 0.38 (n=100);
  mirror cell (expected 0.5): 0.46 (n=200), 0.38 (n=100). Cause UNKNOWN (engine nondeterminism vs shared in-process state between two
  same-class agents). Use +/-0.08 for those cells until a process-isolated rerun settles it.
* Candidate ensemble (B+A nets): all four cells within noise of B-v1 -> not adopted.
* Deck variants under the B-v1 net (100 games, 5 of 8 variants): anti_ogerpon, boss, budew, budew_v2 did not beat base grimmsnarl
  overall (e.g. vs Ogerpon deck 0.43/0.39/0.51/0.40 vs base 0.38-0.53). g4, xero2, xerosic, community not screened.

## Honest conclusion
Nothing measured supports a 1200+ claim. The shipped tarball is B-v1 on the base Grimmsnarl deck; expect a ladder rating near the
~781 already observed (+/-50 noise). The next steps with real expected value are (a) the Kaggle notebook: rating-filtered BC from the
daily episodes datasets, gated by seat-balanced >=300-game tests, (b) a Slowking and strong-Dragapult/Ogerpon proxy opponent set,
(c) a process-isolated rerun of policy-vs-policy cells, (d) submitting >=2 candidates and judging by ladder average over days.
