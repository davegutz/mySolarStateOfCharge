# Decision Tables — SOC_Particle

Auto-generated from `DecisionTables.ods`. The Active-Standby sheet is superseded by Hi-Lo and kept for reference only.

---

## Contents

- [Summary](#summary)
- [Single Hard Faults](#single-hard-faults)
- [Single Soft Faults](#single-soft-faults)
- [Active-Standby Selection (old)](#active-standby-selection-old)
- [Hi-Lo Selection](#hi-lo-selection)

---

## Summary

The first version was Active – Standby. I used it to manage various versions of dual high ranging
sensors. The strategy for selection was the same active standby. Either both sensors were identical
or one was clearly superior by design. I have a lot of experience with active-standby and could
quickly write heuristic algorithms to manage this. The heart of the decision tables have just a few
pithy calculations. With the various configurations I began with a non-amplified version of dubious
precision that I called ‘NOA’ for non-amplified. The other I called ‘AMP’. When I went to two
amplified sensors the standby I called ‘NOA’. For active – standby there are two states: normal and
failed. The failed condition was called ‘latched_fail.’ The basic assumption is that the failures
are annunciated so user can fix right away. This reduced exposure to failure operation results in a
very high reliability system. Along with this goes non-latching low level failures but latching high
level decisions. There is a bug in this: ib_fa values are not annunciated and the failed signals are
still used in logic. In the Hi-Lo I changed the name of bare variables to be ib_fa and added
appropriate logic to the hard fault logic. I only ever ran this logic on a device in ‘Fake Fault’
mode so none of this mattered anyway. I was learning.


I decided to pursue a Hi – Lo strategy in an attempt to prefer a very high gain, low ranging sensor
for accuracy and the existing relatively low gain, high ranging sensor already used. I continued to
use the same names, with ‘AMP’ referring to the high gain / low range and ‘NOA’ referring to the low
gain / high range. The ‘NOA’ is inherently less preferred but now necessary because the solar system
occasionally deals with high currents. I split e_wrap logic to help decide. I kept the original
e_wrap logic using the selected configuration in case it was useful to detect a bad final selection.
I wanted to keep the high reliability with the same annunciation concept. This means the same
latching / non-latching behavior and same low-level Hard Fault and Bare Detection. This started out
also in ‘Fake Fault’ mode to learn.


Hard Faults’ and ‘Bare Detection’ are the same between these two broad strategies.


Perhaps if I ever truly deployed this device I would turn off ‘Fake Fault’ mode.


All parameters are unlatched unless ‘latch’ specifically in the name.


Hard Faults take precedence over reasoned faults in older ‘Active-Standby’ or newer ‘Hi-Lo’


Soft Faults are used in reasoning

---

## Single Hard Faults

### Fault::vc_check

| # | Vc_ < VC_BARE_DETECTED | abs(Ishunt_cal)>=IB_ABS_MAX_?… | vc_fa LATCH |
| - | ---------------------- | ------------------------------ | ----------- |
| 1 | T                      | T                              | T           |

### Shunt::convert Bare

| # | HDWE_BARE | HDWE_ADS1013_AMP_NOA | Vc_ < VC_BARE_DETECTED | using_opamp | Shunt→bare_shunt_ | ib_???_fa LATCH |
| - | --------- | -------------------- | ---------------------- | ----------- | ----------------- | --------------- |
| 1 | T         | T                    | ·                      | ·           | T                 | T               |
| 2 | F         | T                    | ·                      | ·           | F                 | ·               |
| 3 | T         | F                    | ·                      | ·           | F                 | ·               |
| 4 | F         | F                    | T                      | T           | T                 | T               |

### Fault::shunt_check

| # | HDWE_BARE | vc_fa | Shunt→bare_shunt_ | abs(Ishunt_cal)>=IB_ABS_MAX_?… | abs(Ishunt_cal)>=NOM_UNIT_CAP… | ib_???_fa LATCH | Comment |
| - | --------- | ----- | ----------------- | ------------------------------ | ------------------------------ | --------------- | ------- |
| 1 | F         | ·     | T                 | ·                              | ·                              | T               |         |
| 2 | F         | ·     | ·                 | T                              | ·                              | T               |         |
| 3 | T         | ·     | ·                 | ·                              | T                              | T               |         |
| 4 | ·         | T     | ·                 | ·                              | ·                              | T               |         |
| 5 | ·         | ·     | ·                 | ·                              | ·                              | F               | Default |

### Fault::vb_check

| # | vb <= VB_MIN &&ib_hdwe*nP > I… | vb>=VB_MAX | vb_fa LATCH | Comment              |
| - | ------------------------------ | ---------- | ----------- | -------------------- |
| 1 | T                              | ·          | T           | go to Active-Standby |
| 2 | ·                              | T          | T           |                      |
| 3 | ·                              | ·          | F           | Default              |

### Fault::tb_check

| # | Tb>=TB+<OM | Tb>=TB_MAX | Persisted tb_stale_flt | tb_fa | Comment |
| - | ---------- | ---------- | ---------------------- | ----- | ------- |
| 1 | T          | ·          | ·                      | T     |         |
| 2 | ·          | T          | ·                      | T     |         |
| 3 | ·          | ·          | T                      | T     |         |
| 4 | ·          | ·          | ·                      | F     | Default |

### disconnect

| # | quiet(ib_noa + ib_amp) < quie… | ib_dscn_fa | Comment |
| - | ------------------------------ | ---------- | ------- |
| 1 | T                              | T          |         |
| 2 | F                              | F          | Default |

> Notes:
> vb_fail = vb_sel_stat==0 || vb_fa = wrap_vb_fa || vb_fa


---

## Single Soft Faults

### Fault::ib_wrap

| # | sat | (voc_soc – voc_stat) > ewhi_t… | voc_soc() - voc_amp >= ewhi_t… | voc_soc() - voc_amp <= ewlo_t… | voc_soc() - voc_noa >= ewhi_t… | voc_soc() - voc_noa <= ewlo_t… | wrap_lo_m_fa | wrap_hi_m_fa | wrap_lo_n_fa | wrap_hi_n_fa | e_wrap_fa |
| - | --- | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ | ------------ | ------------ | ------------ | ------------ | --------- |
| 1 | F   | T                              | ·                              | ·                              | ·                              | ·                              | ·            | ·            | ·            | ·            | T         |
| 2 | ·   | ·                              | T                              | ·                              | ·                              | ·                              | T            | ·            | ·            | ·            | ·         |
| 3 | ·   | ·                              | ·                              | T                              | ·                              | ·                              | ·            | T            | ·            | ·            | ·         |
| 4 | ·   | ·                              | ·                              | ·                              | T                              | ·                              | ·            | ·            | T            | ·            | ·         |
| 5 | ·   | ·                              | ·                              | ·                              | ·                              | T                              | ·            | ·            | ·            | T            | ·         |

### Fault::ib_wrap

| # | reset_all_faults_ | ib_diff_fa | wrap_m_and_n_fa | wrap_vb_fa | vb_sel_stat_ | latched_fail_ | Comment                 |
| - | ----------------- | ---------- | --------------- | ---------- | ------------ | ------------- | ----------------------- |
| 1 | T                 | ·          | ·               | F          | 1            | F             |                         |
| 2 | ·                 | F          | T               | T          | 0            | T             | Isolated to vb. Latches |

### Fault::ib_diff

| # | reset_all_faults_ | ib_lo_active_ | abs(ib_amp – ib_noa) >= IBATT… | ib_diff_fa |
| - | ----------------- | ------------- | ------------------------------ | ---------- |
| 1 | T                 | ·             | ·                              | F          |
| 2 | ·                 | T             | T                              | T          |


---

## Active-Standby Selection (old)

### Reset

| # | latched_fail_ | ap.fake_fault | sp.ib_force (i_f) | reset_all_faults_ | sp.mod_ib | sp.mod_vb | ib_sel_stat_last_ | wrap_vb_fa_ | vb_sel_stat_ | ib_sel_stat_ | latched_fail_ |
| - | ------------- | ------------- | ----------------- | ----------------- | --------- | --------- | ----------------- | ----------- | ------------ | ------------ | ------------- |
| 1 | x             | x             | x                 | T                 | x         | x         | x                 | F           | 1            | i_f          | F             |

### Fake Faults

| # | ap.fake_fault | ib_sel_stat_ | latched_fail_ |
| - | ------------- | ------------ | ------------- |
| 1 | T             | i_f          | F             |

### ib_decision_active_standby

| #  | Section      | latched_fail_ | ap.fake_fault | sp.ib_force (i_f) | reset_all_faults_ | sp.mod_ib | ib_sel_stat_last_ | ib_amp_fa = ib_amp_bare_ | ib_noa_fa = ib_noa_bare_ | ib_diff_fa | vb_sel_stat_last_ | e_wrap_fa | cc_diff_fa | ib_decision_ | ib_sel_stat_ | latched_fail_ | red_loss() | Comment                                                               |
| -- | ------------ | ------------- | ------------- | ----------------- | ----------------- | --------- | ----------------- | ------------------------ | ------------------------ | ---------- | ----------------- | --------- | ---------- | ------------ | ------------ | ------------- | ---------- | --------------------------------------------------------------------- |
| 1  |              | T             | ·             | ·                 | ·                 | ·         | ·                 | ·                        | ·                        | ·          | ·                 | ·         | ·          | 10           | ·            | ·             | ·          | latch is a latch is a latch iff !latched_fail                         |
| 2  |              | ·             | F             | ·                 | ·                 | ·         | ·                 | ·                        | ·                        | ·          | ·                 | ·         | ·          | 0            | 1            | F             | ·          |                                                                       |
| 3  |              | ·             | ·             | ·                 | ·                 | ·         | ·                 | T                        | T                        | ·          | ·                 | ·         | ·          | 1            | 0            | T             | T          |                                                                       |
| 4  |              | ·             | ·             | 1                 | ·                 | ·         | ·                 | F                        | ·                        | ·          | ·                 | ·         | ·          | 2            | 1            | T             | ·          | Forcing ib to one loses redundancy                                    |
| 5  |              | ·             | ·             | ·                 | F                 | F         | -1                | ·                        | F                        | ·          | ·                 | ·         | ·          | 3            | -1           | T             | T          | Cannot reset except by hard reset or mod_ib set. Forces user to think |
| 6  |              | ·             | ·             | -1                | F                 | ·         | ·                 | ·                        | F                        | ·          | ·                 | ·         | ·          | 4            | -1           | T             | T          | Forcing ib to one loses redundancy                                    |
| 7  | auto section | ·             | ·             | 0                 | ·                 | ·         | ·                 | T                        | F                        | ·          | ·                 | ·         | ·          | 5            | -1           | T             | T          | ib_amp is primary in active standby process                           |
| 8  | auto section | ·             | ·             | 0                 | ·                 | ·         | ·                 | ·                        | ·                        | T          | 1                 | T         | ·          | 6            | -1           | T             | T          | Isolated to ib_amp                                                    |
| 9  | auto section | ·             | ·             | 0                 | ·                 | ·         | ·                 | ·                        | ·                        | T          | ·                 | ·         | T          | 7            | -1           | T             | T          | Isolated to ib_amp                                                    |
| 10 | auto section | ·             | ·             | -1                | ·                 | ·         | 0                 | ·                        | ·                        | ·          | ·                 | ·         | ·          | 8            | 0            | T             | T          |                                                                       |
| 11 | auto section | ·             | ·             | -1                | ·                 | ·         | 1                 | ·                        | ·                        | ·          | ·                 | ·         | ·          | 8            | 1            | T             | T          |                                                                       |
| 12 | auto section | ·             | ·             | 1                 | ·                 | ·         | 0                 | ·                        | ·                        | ·          | ·                 | ·         | ·          | 8            | 0            | T             | T          |                                                                       |
| 13 | auto section | ·             | ·             | 1                 | ·                 | ·         | -1                | ·                        | ·                        | ·          | ·                 | ·         | ·          | 8            | -1           | T             | T          |                                                                       |
| 14 | auto section | ·             | ·             | 0                 | ·                 | ·         | x                 | ·                        | ·                        | ·          | ·                 | ·         | ·          | 9            | ·            | ·             | F          | Not reachable but here for completeness to avoid indecision           |

### red_loss

| # | ap.fake_fault | sp.ib_force (i_f) | ib_sel_stat_last_ | ib_amp_fa = ib_amp_bare_ | ib_noa_fa = ib_noa_bare_ | ib_diff_fa | vb_fa | red_loss() | Comment |
| - | ------------- | ----------------- | ----------------- | ------------------------ | ------------------------ | ---------- | ----- | ---------- | ------- |
| 1 | ·             | ·                 | 0                 | ·                        | ·                        | ·          | ·     | T          |         |
| 2 | ·             | ·                 | 1                 | ·                        | ·                        | ·          | ·     | T          |         |
| 3 | F             | -1                | ·                 | ·                        | ·                        | ·          | ·     | T          |         |
| 4 | F             | 1                 | ·                 | ·                        | ·                        | ·          | ·     | T          |         |
| 5 | ·             | ·                 | ·                 | ·                        | ·                        | T          | ·     | T          |         |
| 6 | ·             | ·                 | ·                 | T                        | ·                        | ·          | ·     | T          |         |
| 7 | ·             | ·                 | ·                 | ·                        | T                        | ·          | ·     | T          |         |
| 8 | ·             | ·                 | ·                 | ·                        | ·                        | ·          | T     | T          |         |
| 9 | ·             | ·                 | ·                 | ·                        | ·                        | ·          | ·     | F          | Default |

### bms off

| # | temp_c < chem_.low_t | voc_stat < chem_.vb_down | bms_off |
| - | -------------------- | ------------------------ | ------- |
| 1 | T                    | ·                        | T       |
| 2 | ·                    | T                        | T       |


---

## Hi-Lo Selection

### Reset

| # | latched_fail_ | ap.fake_fault | sp.ib_force (i_f) | reset_all_faults_ | sp.mod_ib | sp.mod_vb | ib_sel_stat_last_ (ibl) | wrap_vb_fa_ | vb_sel_stat_ | ib_choice_(-1=noa,0=def,1=amp… | latched_fail_ |
| - | ------------- | ------------- | ----------------- | ----------------- | --------- | --------- | ----------------------- | ----------- | ------------ | ------------------------------ | ------------- |
| 1 | x             | x             | x                 | T                 | x         | x         | x                       | F           | 1            | i_f                            | F             |

### Fault::ib_select_decision_hi_lo

| #  | Section      | latched_fail_ | sp.ib_force (i_f) | reset_all_faults_ | sp.mod_ib | sp.mod_vb | ib_choice_(-1=noa,0=def,1=amp… | ib_amp_fa | ib_noa_fa | ib_diff_fa | vb_sel_stat_last_ (vbl) | wrap_m_fa | wrap_n_fa | cc_diff_fa | ib_decision_ | ib_choice_(-1=noa,0=def,1=amp… | latched_fail_ | red_loss() | Comment                                                     |
| -- | ------------ | ------------- | ----------------- | ----------------- | --------- | --------- | ------------------------------ | --------- | --------- | ---------- | ----------------------- | --------- | --------- | ---------- | ------------ | ------------------------------ | ------------- | ---------- | ----------------------------------------------------------- |
| 1  |              | T             | ·                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | ·          | last         | ·                              | ·             | ·          | must reset (Rf) or reinstall and set nominal                |
| 2  |              | ·             | ·                 | ·                 | ·         | ·         | ·                              | T         | T         | ·          | ·                       | ·         | ·         | ·          | 1            | -2                             | T             | ·          |                                                             |
| 3  |              | ·             | 1                 | ·                 | ·         | ·         | ·                              | F         | ·         | ·          | ·                       | ·         | ·         | ·          | 2            | 1                              | T             | ·          | Forcing ib to one loses redundancy                          |
| 4  |              | ·             | -1                | F                 | ·         | ·         | ·                              | ·         | F         | ·          | ·                       | ·         | ·         | ·          | 3            | -1                             | T             | ·          | Forcing ib to one loses redundancy                          |
| 5  | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | T         | F         | ·          | ·                       | ·         | ·         | ·          | 4            | -1                             | T             | ·          | still ‘works’                                               |
| 6  | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | F         | T         | ·          | ·                       | ·         | ·         | ·          | 5            | 1                              | T             | ·          | still ‘works’                                               |
| 7  | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | T          | 1                       | T         | F         | ·          | 6            | -1                             | T             | ·          | ampHiFail                                                   |
| 8  | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | T          | 1                       | F         | T         | ·          | 7            | 1                              | T             | ·          | lose accy of tracking high current. NoaHiFail               |
| 9  | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | T          | 1                       | T         | T         | ·          | 8            | 0                              | F             | ·          | keep trying; ambiguous                                      |
| 10 | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | vbl                     | ·         | ·         | ·          | 0            | ibl                            | vbl           | ·          | Default                                                     |
| 11 | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | T          | ·                       | ·         | ·         | T          | 10           | -1                             | T             | ·          | still ‘works’                                               |
| 12 | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | vbl                     | ·         | ·         | ·          | 0            | ibl                            | vbl           | ·          | Default                                                     |
| 13 | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | T          | 12           | 0                              | F             | ·          | keep trying; ambiguous                                      |
| 14 | auto section | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | vbl                     | ·         | ·         | ·          | 0            | ibl                            | vbl           | ·          | Default                                                     |
| 15 | ---->        | ·             | -1                | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | ·          | 14           | 0                              | T             | ·          |                                                             |
| 16 | ---->        | ·             | -1                | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | ·          | 14           | 1                              | T             | ·          | Forcing ib loses redundancy                                 |
| 17 | ---->        | ·             | 1                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | ·          | 14           | 0                              | T             | ·          |                                                             |
| 18 | ---->        | ·             | 1                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | ·          | 14           | -1                             | T             | ·          | Forcing ib loses redundancy                                 |
| 19 | ---->        | ·             | 0                 | ·                 | ·         | ·         | ·                              | ·         | ·         | ·          | ·                       | ·         | ·         | ·          | 15           | ·                              | F             | ·          | Not reachable but here for completeness to avoid indecision |
| 20 | ---->        | ·             | ·                 | ·                 | F         | F         | 1                              | ·         | ·         | ·          | x                       | ·         | ·         | ·          | ·            | ·                              | ·             | 1          | if modeling get bogus display without the FF                |
| 21 | ---->        | ·             | ·                 | ·                 | F         | F         | -1                             | ·         | ·         | ·          | x                       | ·         | ·         | ·          | ·            | ·                              | ·             | 1          |                                                             |
| 22 | ---->        | ·             | ·                 | ·                 | F         | F         | x                              | ·         | ·         | ·          | 0                       | ·         | ·         | ·          | ·            | ·                              | ·             | 1          |                                                             |
| 23 | ---->        | ·             | ·                 | ·                 | ·         | ·         | 0                              | ·         | ·         | ·          | 1                       | ·         | ·         | ·          | ·            | ·                              | ·             | 0          |                                                             |

### Fault::vb_select_decision_hi_lo

| # | latched_fail_ | ib_diff_fa | wrap_m_fa | wrap_n_fa | vb_fa | wrap_vb_fa_ | vb_sel_stat_ | latched_fail_ |
| - | ------------- | ---------- | --------- | --------- | ----- | ----------- | ------------ | ------------- |
| 1 | T             | ·          | ·         | ·         | ·     | ·           | last         | ·             |
| 2 | ·             | ·          | ·         | ·         | T     | ·           | 0            | T             |
| 3 | ·             | F          | T         | T         | ·     | T           | 0            | T             |
| 4 | ·             | T          | ·         | ·         | ·     | ·           | vbl          | ·             |

### bms off

| # | temp_c < chem_.low_t | voc_stat < chem_.vb_down | bms_off |
| - | -------------------- | ------------------------ | ------- |
| 1 | T                    | ·                        | T       |
| 2 | ·                    | T                        | T       |

### e_wrap

| # | wrap_m_fa | wrap_n_fa | e_wrap_fa | Comment |
| - | --------- | --------- | --------- | ------- |
| 1 | T         | T         | T         |         |
| 2 | ·         | ·         | F         | Default |

### soft_reset

| # | ib_choice_(-1=noa,0=def,1=amp… | ib_choice != ib_choice_last_ | cmd_reset pulse |
| - | ------------------------------ | ---------------------------- | --------------- |
| 1 | -1                             | T                            | T               |
| 2 | ·                              | ·                            | F               |

> Dm-50


---
