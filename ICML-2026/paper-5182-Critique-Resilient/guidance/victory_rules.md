This document explains what to count, for a given entry, as a victory for Alice, a victory for Bob, or neither.

# Reading Judgments

A judgment is an evaluation of a critique or an ill-posedness claim.

Precedence:
1. If there is a human judgment, it takes precedence over automated judgments
2. If there are multiple human judgments which are not the same, throw an error
3. If the automated judgments are not unanimous, ignore but log an error
4. If the automated judgments are unanimous, use it

Interpreting an ill-posedness claim judgment:
1. claimant_wins => Alice wins
2. defender_wins_incorrect, wrong_problem => Bob wins
3. All others => Neither

Interpreting a critique claim:
1. claimant_wins => Alice wins
2. defender_wins_incorrect, defender_wins_minor, wrong_problem => Bob wins
3. All others => Neither

# From Judgments to Victories

For a lot of analyses, we need to determine who "won" a certain game.
There are only three possible outcomes: Alice wins, Bob wins, or dropped (which is different from a tie).

A game is simply the result of a trace (e.g. question => answer => critique, or question => answer => critique => debate => evaluation, or question => answer (with an ill-posedness claim) => debate => evaluation)
Let's go over the protocol and check all cases

1. Alice makes a question
2. Alice answers her own question
3. Bob critiques Alice's question
  3a. Bob's critique says it's correct => continue with the protocol
  3b. Bob's critique says it's incorrect* => if Bob wins the claim, drop this game as well as all games that rely on this question. Otherwise continue
4. Bob answers Alice's question
  4a. Bob actually provides the answer => continue with the protocol
  4b. Bob claims it's ill-posed => if Bob wins the claim, drop this game as well as all games that rely on this question. Otherwise continue
  4c. Bob fails to answer => Alice wins
5. Alice critiques Bob's answer
  5a. Alice's critique says it's correct => Bob wins
  5b. Alice's critique says it's incorrect* => If Alice wins the claim, Alice wins. If Bob wins the claim, Bob wins. Otherwise, drop this game

* Including insufficient and obscure

In case of any unaccounted-for failures or missing entry, drop the game
