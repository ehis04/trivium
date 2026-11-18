# Findings: tactical profiles that beat their strength

Tournament pathways. All numbers come from the real build of
30 October 2026 (`check_report.md`, `finding_gate`, `tactical_profile_*` tables); team level only.

## The answer

A tactical profile scores each team on three definitions, standardised within its tournament:

| Score | Built from | Higher means |
|---|---|---|
| Dominant | dominance: mean of contests won, field tilt, share of non-penalty xG | more of the match won |
| Efficient | conversion (xG per unit of field tilt), security (xG conceded per opponent entry) | dominance turned into chances, few conceded |
| Resourceful | defensive actions per point of dominance; men also high-speed running without the ball per point | dominance at a lower cost |

Performance is points above the expectation from senior Elo (a leave-team-out ordered logit on the Elo gap), so a
strong team does not look good only for being strong.

**Men (48 teams).** The dominant and resourceful scores pass every test of the slide gate:

| Score | Rank correlation with points above expectation per match | Repeatable (split-half, 0.5 bar) | First matches predict the rest |
|---|---|---|---|
| Dominant | +0.43 (95% +0.20 to +0.64) | 0.79, yes | +0.32 (+0.02 to +0.54), yes |
| Resourceful | +0.41 (+0.19 to +0.61) | 0.83, yes | +0.31 (+0.02 to +0.55), yes |
| Efficient | +0.50 (+0.29 to +0.69) | 0.44, no | +0.31 (+0.02 to +0.55) |

**Women (24 teams).** Efficient +0.49 (+0.14 to +0.80) but not repeatable (0.33); dominant +0.12 and resourceful
+0.24, same direction as the men, not clear.

**By profile (men).** Dominant, efficient and resourceful: 23 teams, +0.22 points per match above expectation
(-0.01 to +0.44), 15 of 23 above it. Neither dominant nor efficient: 15 teams, 1 above expectation (costly: 11
teams, -0.61, -0.89 to -0.31; resourceful: 4 teams, -0.83). Women, dominant, efficient and resourceful: 9 teams,
+0.35 (-0.09 to +0.78), 7 of 9 above.

**Ireland.** Dominant, efficient and resourceful (dominant +0.3, efficient +0.8, resourceful +0.3 standard
deviations): 8 points against 4.24 expected (+3.76), round of 16 (1.35 stages beyond the expected stage).
Panama 1-4 Ireland: Ireland 37% field tilt and 48% dominance, but 63% of the chances (1.98 xG against 1.14).

## How sure

- Other definitions (men): every variant keeps the sign. Dominant from chances share alone +0.51, field tilt alone
  +0.27, contests alone +0.06 (the chances part carries it). Efficient from conversion alone +0.40, security alone
  +0.46. Resourceful from defensive work alone +0.33, running without the ball alone +0.42.
- Group-stage matches only: dominant +0.46, efficient +0.50, resourceful +0.44 (all clear).
- Without the strongest quarter of teams by Elo: +0.49, +0.47, +0.46 (all clear).
- Profile labels: 99% unchanged when any one team is left out (women 97%); 69% unchanged with the median instead
  of the mean as the threshold (women 83%).
- Strength baseline: all 72 teams matched to Elo; the higher-rated team won 41 of 58 decisive men's group matches
  and 26 of 33 women's. On senior World Cups the expectation model beats base rates in 9 of 10 editions.

## The three points (A, B and C test the answer)

Teams above against below their Elo expectation, in standard deviations, after accounting for strength
(`profile_comparison`; grouped by `models.answer.point_verdicts`). "Clear" means the 95% interval excludes zero.

**Dominance: control pays off through the chances, not the duels (clear in both tournaments).**
Chances share men +1.04 (+0.59 to +1.48), women +0.91 (+0.13 to +1.62); field tilt men +0.73; overall dominance
men +0.89. Contests won: no clear difference in either tournament (men +0.34, women +0.53).

**Resourcefulness: it is not how much a team spends but what each point of control costs (clear in both).**
- Cost per point of dominance: defensive work -0.68 in both tournaments (men -1.19 to -0.24, women -1.43 to
  -0.04); high-speed running without the ball men -0.62 (-1.16 to -0.21).
- Spent: running (total distance, high-speed running, sprinting), pressing (pressures, per minute of opponent
  possession), defensive actions and squad rotation (minutes outside the most-used 11, starters changed, spread of
  running): no clear difference in either tournament. The exception: men's teams that beat their strength had more
  of the ball (possession share +0.55, +0.02 to +1.06), only just clear; women not clear.

**Efficiency: what the control produced (clear in both).**
Security (xG conceded per opponent entry) men -0.94 (-1.39 to -0.49), women -1.20 (clear after accounting for
strength); top-2 players' share of xG + xA men -0.75 (chances spread across more players); spread of chances
across zones men +0.55 (after accounting for strength). Conversion (xG per unit of field tilt): no clear difference.

Playing-style clusters (possession, pressing, FIFA phases) were tried first: only one small men's group (6 teams)
stood out, below expectation.

## Formations (from tracking, Emmanuel Gyamfi's method)

Average outfield positions per 5-minute window, with and without the ball (101 men's matches), grouped into lines
by Jenks natural breaks on depth, the number of lines by the goodness of variance fit. Without the ball the most
common shapes are 4-2-3-1, 4-4-2, 4-3-3 and 5-4-1. The detected back line agrees with the StatsBomb listed
formation in 70% of team-matches; a listed back three is often detected as a back five (wing-backs drop). Shapes
change from window to window (the main shape covers about a third of a match's windows), so formations describe
shape, not a fixed label.

## Data and what was left out

- StatsBomb: men 104 matches, 307,480 events, 48 teams; women 52 matches, 159,256 events, 24 teams; goals reconcile
  with every score.
- Men: FIFA physical files and broadcast tracking each cover 103 of 104 matches (M27 has no physical file, M73 no
  tracking). Two matches (M81 Ireland v Canada, 70%; M95, 39%) had more than 30% of player positions filled in by
  the provider and are left out of the tracking measures. Tracking distance is within 2% of FIFA for every team;
  188 of 206 halves sync events to tracking within 5 yards (median 3.5).
- Penalty shootouts are left out of performance; shootout results count for progression.

## Limits

- Senior Elo stands in for under-17 strength.
- Few teams per profile: intervals are wide; this is association, not cause.
- Efficiency is not repeatable from match to match; the women's tournament (24 teams, no tracking) is not clear.
- Later matches only include teams still in the tournament (survivor effect); the out-of-sample test uses each
  team's own first and later matches.

## Questions a reader may ask

- **Is this just strong teams being good?** No: every outcome is points above the team's own Elo expectation, and
  the result holds with the strongest quarter of teams left out.
- **Is it luck?** The scores repeat from match to match, and measured in a team's first matches they predict its
  later results.
- **Does the definition matter?** Every variant keeps the sign; the chances part of dominance carries the most.
- **Why not just territory?** Field tilt alone goes with beating expectation more weakly (+0.27) than the share of
  the chances (+0.51); Ireland beat Panama 4-1 with 37% field tilt and 63% of the chances.
- **What would a coach do?** After each group match, check the three scores: dominant at a low cost is the profile
  that goes further than strength predicts; if dominance costs too much running and defensive work, change how
  and where the team wins the ball before the knockouts.
