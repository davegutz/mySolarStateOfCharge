<style>pre { white-space: pre; overflow-x: auto; }</style>

# Decision Tables — SOC_Particle

## Contents

- [Single Hard Faults](#single-hard-faults)
- [Soft Faults](#soft-faults)
- [Vb Selection](#vb-selection)
- [Ib Active-Standby Selection](#ib-active-standby-selection)
- [Ib Hi-Lo Selection](#ib-hi-lo-selection)
- [Annunciation](#annunciation)

---

## Summary

Auto-generated from `DecisionTables.ods`. The Active-Standby sheet is kept for
reference; Hi-Lo is the active strategy.

All parameters are unlatched unless `latch` specifically appears in the name.
Hard Faults take precedence over Soft Faults. Soft Faults are used in reasoning.
`·` = don't-care / any value.  `||` separates condition inputs from results.

**Abbreviations**

| Abbrev | Meaning |
| ------ | ------- |
| Rf     | `reset_all_faults` |
| Ff     | `ap.fake_fault` |
| si     | `sp.ib_force` |
| ibl    | last good ib_sel value |
| vbl    | last good vb_sel value |


---

## Single Hard Faults

### Fault::Tb_check

```text
| # | Tb       | Tb       | Persisted    || Tb_fa |
|   | <=TB_MIN | >=TB_MAX | tb_stale_flt ||       |
| - | -------- | -------- | ------------ || ----- |
| 1 | T        | ·        | ·            || T     |
| 2 | ·        | T        | ·            || T     |
| 3 | ·        | ·        | T            || T     |
```

### Fault::vb_check

```text
| # | Ib_hdwe     | Vb        | vb       || vb_fa_lt |
|   | > IB_MIN_UP | <= vb_min | >=VB_MAX || LATCH |
| - | ----------- | --------- | -------- || ----- |
| 1 | T           | T         | ·        || T     |
| 2 | ·           | ·         | T        || T     |
```

### Fault::vc_check

```text
| # | Vc <= vc_min    || vc_fa |
|   | or Vc >= vc_max || LATCH |
| - | --------------- || ----- |
| 1 | T               || T     |
```

### Shunt::convert Bare

```text
| # | Vc <= vc_min    || using_opamp | Shunt→bare_shunt_ | ib_???_fa |
|   | or Vc >= vc_max ||             |                   | LATCH     |
| - | --------------- || ----------- | ----------------- | --------- |
| 1 | T               || T           | T                 | T         |
```

### Fault::shunt_check

```text
| # | Shunt→bare_shunt_ | vc_fa | abs(Ishunt_cal)  || ib_???_fa |
|   |                   |       | >=IB_ABS_MAX_??? || LATCH     |
| - | ----------------- | ----- | ---------------- || --------- |
| 1 | T                 | ·     | ·                || T         |
| 2 | ·                 | ·     | T                || T         |
| 3 | ·                 | T     | ·                || T         |
```


---

## Soft Faults

### Fault::ib_diff

```text
| # | ib_diff           | ib_diff           || ib_diff_hi_fa | ib_diff_lo_fa | ib_diff_fa |
|   | >= IB_DIFF_HI_FLT | <= IB_DIFF_LO_FLT ||               |               |            |
| - | ----------------- | ----------------- || ------------- | ------------- | ---------- |
| 1 | T                 | ·                 || T             | ·             | T          |
| 2 | ·                 | T                 || ·             | T             | T          |
```

### Fault::ib_wrap; ib section

```text
| #  | HDWE_IB_HI_LO | sat | ib_really_quiet | ib_diff_fa | dv_dyn_active     | dv_dyn_active     | dv_dyn_amp      | dv_dyn_amp      | dv_dyn_noa      | dv_dyn_noa      || wrap_hi_m_fa | wrap_lo_m_fa | wrap_hi_n_fa | wrap_lo_n_fa | wrap_m_and_n_fa | wrap_lo_fa |
|    |               |     |                 |            | > ewhi_thr_active | < ewlo_thr_active | >= ewhi_thr_amp | <= ewlo_thr_amp | >= ewhi_thr_noa | <= ewlo_thr_noa ||              |              |              |              |                 |            |
| -- | ------------- | --- | --------------- | ---------- | ----------------- | ----------------- | --------------- | --------------- | --------------- | --------------- || ------------ | ------------ | ------------ | ------------ | --------------- | ---------- |
| 1  | T             | ·   | ·               | ·          | ·                 | ·                 | T               | ·               | ·               | ·               || ·            | ·            | ·            | ·            | ·               | ·          |
| 2  | T             | ·   | ·               | ·          | ·                 | ·                 | ·               | T               | ·               | ·               || ·            | T            | ·            | ·            | ·               | ·          |
| 3  | T             | ·   | ·               | ·          | ·                 | ·                 | ·               | ·               | T               | ·               || T            | ·            | ·            | ·            | ·               | ·          |
| 4  | T             | ·   | ·               | ·          | ·                 | ·                 | ·               | ·               | ·               | T               || ·            | ·            | ·            | T            | ·               | ·          |
| 5  | T             | F   | T               | T          | ·                 | ·                 | T               | ·               | T               | ·               || T            | ·            | T            | ·            | T               | ·          |
| 6  | T             | ·   | ·               | ·          | ·                 | ·                 | ·               | T               | ·               | T               || ·            | T            | ·            | T            | T               | T          |
| 7  | F             | F   | ·               | ·          | T                 | ·                 | ·               | ·               | ·               | ·               || ·            | ·            | T            | ·            | ·               | ·          |
| 8  | F             | ·   | ·               | ·          | ·                 | T                 | ·               | ·               | ·               | ·               || ·            | ·            | ·            | ·            | ·               | T          |
| 9  | F             | ·   | ·               | ·          | ·                 | ·                 | ·               | T               | ·               | ·               || ·            | ·            | ·            | ·            | ·               | ·          |
| 10 | F             | ·   | ·               | ·          | ·                 | ·                 | T               | ·               | ·               | ·               || ·            | T            | ·            | ·            | ·               | ·          |
| 11 | F             | ·   | ·               | ·          | ·                 | ·                 | ·               | ·               | ·               | T               || T            | ·            | ·            | ·            | ·               | ·          |
| 12 | F             | ·   | ·               | ·          | ·                 | ·                 | ·               | ·               | T               | ·               || ·            | ·            | ·            | T            | ·               | ·          |
```

### Fault::ib_wrap; vb section

```text
| # | ib_really_quiet | ib_diff_fa | wrap_m&n || wrap_vb_fa | vb_sel_stat |
|   |                 |            |          || LATCH      |             |
| - | --------------- | ---------- | -------- || ---------- | ----------- |
| 1 | T               | F          | F        || T          | 1           |
```

> **Notes:**
>
> ib_diff = ib_amp – ib_noa
>
> voc_* = vb – dv_dyn_*
>
> dv_dyn_* = voc_soc – voc_*
>
> wrap_m&n = wrap_m_and_n_fa


---

## Vb Selection

### vb_selection

```text
| # | vb_fa_lt   | wrap_vb_fa || vb_sel_stat | Comment                                                                                                                           |
|   | LATCHED |            ||             |                                                                                                                                   |
| - | ------- | ---------- || ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1 | T       | ·          || 0           | Simplex Vb always available without modification. But used carefully e.g. low Vb (potentially failed) is confirmed with Ib quiet. |
| 2 | ·       | T          || 0           | ·                                                                                                                                 |
| 3 | ·       | ·          || 1           | Default                                                                                                                           |
```


---

## Ib Active-Standby Selection

### ib_decision_active_standby

```text
| #  | ib_force | mod_ib | ib_sel_stat_last | ib_amp_fa = | ib_noa_fa = | ib_diff_fa | vb_sel_stat_last | e_wrap_fa | cc_diff_fa || ib_decision | ib_sel_stat | latched_fail | red_loss() | Comment                                                               |
|    | (si)     |        |                  | ib_amp_bare | ib_noa_bare |            |                  |           |            ||             |             |              |            |                                                                       |
| -- | -------- | ------ | ---------------- | ----------- | ----------- | ---------- | ---------------- | --------- | ---------- || ----------- | ----------- | ------------ | ---------- | --------------------------------------------------------------------- |
| 1  | ·        | ·      | ·                | ·           | ·           | ·          | ·                | ·         | ·          || 10          | ·           | ·            | ·          | latch is a latch is a latch iff !latched_fail                         |
| 2  | ·        | ·      | ·                | T           | T           | ·          | ·                | ·         | ·          || 1           | 0           | T            | T          | ·                                                                     |
| 3  | 1        | ·      | ·                | F           | ·           | ·          | ·                | ·         | ·          || 2           | 1           | T            | ·          | Forcing ib to one loses redundancy                                    |
| 4  | ·        | F      | -1               | ·           | F           | ·          | ·                | ·         | ·          || 3           | -1          | T            | T          | Cannot reset except by hard reset or mod_ib set. Forces user to think |
| 5  | -1       | ·      | ·                | ·           | F           | ·          | ·                | ·         | ·          || 4           | -1          | T            | T          | Forcing ib to one loses redundancy                                    |
| 6  | 0        | ·      | ·                | T           | F           | ·          | ·                | ·         | ·          || 5           | -1          | T            | T          | ib_amp is primary in active standby process                           |
| 7  | 0        | ·      | ·                | ·           | ·           | T          | 1                | T         | ·          || 6           | -1          | T            | T          | Isolated to ib_amp                                                    |
| 8  | 0        | ·      | ·                | ·           | ·           | T          | ·                | ·         | T          || 7           | -1          | T            | T          | Isolated to ib_amp                                                    |
| 9  | -1       | ·      | 0                | ·           | ·           | ·          | ·                | ·         | ·          || 8           | 0           | T            | T          | ·                                                                     |
| 10 | -1       | ·      | 1                | ·           | ·           | ·          | ·                | ·         | ·          || 8           | 1           | T            | T          | ·                                                                     |
| 11 | 1        | ·      | 0                | ·           | ·           | ·          | ·                | ·         | ·          || 8           | 0           | T            | T          | ·                                                                     |
| 12 | 1        | ·      | -1               | ·           | ·           | ·          | ·                | ·         | ·          || 8           | -1          | T            | T          | ·                                                                     |
| 13 | 0        | ·      | x                | ·           | ·           | ·          | ·                | ·         | ·          || 9           | ·           | ·            | F          | Not reachable but here for completeness to avoid indecision           |
| 14 | ·        | ·      | ·                | ·           | ·           | ·          | ·                | ·         | ·          || 0           | 1           | ·            | ·          | Default                                                               |
```


---

## Ib Hi-Lo Selection

### ib_decision_hi_lo()

```text
| #  | ib_force | ib_amp_fa | ib_noa_fa | ib_diff_fa | vb_sel_stat_last | wrap_m_fa | wrap_n_fa | cc_diff_fa | ib_choice || ib_decision | ib_choice | latched          | Comment                                                    |
|    | (si)     |           |           |            | (vbl)            |           |           |            | last      ||             |           | (reset to clear) |                                                            |
| -- | -------- | --------- | --------- | ---------- | ---------------- | --------- | --------- | ---------- | --------- || ----------- | --------- | ---------------- | ---------------------------------------------------------- |
| 1  | ·        | T         | T         | ·          | ·                | ·         | ·         | ·          | ·         || 1           | none      | T                | ib not used                                                |
| 2  | 1        | F         | ·         | ·          | ·                | ·         | ·         | ·          | ·         || 2           | amp       | T                | Forcing ib to one loses redundancy                         |
| 3  | -1       | ·         | F         | ·          | ·                | ·         | ·         | ·          | ·         || 3           | noa       | T                | Forcing ib to one loses redundancy                         |
| 4  | ·        | T         | F         | ·          | ·                | ·         | ·         | ·          | ·         || 4           | noa       | T                | still ‘works’                                              |
| 5  | ·        | F         | T         | ·          | ·                | ·         | ·         | ·          | ·         || 5           | amp       | T                | still ‘works’                                              |
| 6  | ·        | ·         | ·         | T          | 1                | T         | F         | ·          | ·         || 6           | noa       | T                | ampHiFail                                                  |
| 7  | ·        | ·         | ·         | T          | 1                | F         | T         | ·          | ·         || 7           | amp       | T                | lose accy of tracking high current. NoaHiFail              |
| 8  | ·        | ·         | ·         | T          | 1                | T         | T         | ·          | ·         || 8           | try       | F                | keep trying; ambiguous                                     |
| 9  | ·        | ·         | ·         | T          | ·                | ·         | ·         | T          | ·         || 10          | noa       | T                | still ‘works’                                              |
| 10 | ·        | ·         | ·         | ·          | ·                | ·         | ·         | T          | ·         || 12          | try       | F                | keep trying; ambiguous                                     |
| 11 | -1       | ·         | ·         | ·          | ·                | ·         | ·         | ·          | !noa      || 14          | try       | F                | keep trying; ambiguous                                     |
| 12 | -1       | ·         | ·         | ·          | ·                | ·         | ·         | ·          | ·         || 14          | amp       | T                | Forcing ib loses redundancy                                |
| 13 | 1        | ·         | ·         | ·          | ·                | ·         | ·         | ·          | !amp      || 14          | try       | F                | keep trying; ambiguous                                     |
| 14 | 1        | ·         | ·         | ·          | ·                | ·         | ·         | ·          | ·         || 14          | noa       | T                | Forcing ib loses redundancy                                |
| 15 | ·        | ·         | ·         | ·          | ·                | ·         | ·         | ·          | ·         || 15          | try       | F                | Not reachable but here for completeness to avoid dead code |
| 16 | ·        | ·         | ·         | ·          | ·                | ·         | ·         | ·          | ·         || 0           | try       | ·                | Default                                                    |
```


---

## Annunciation - GUI Plot

### serial_display()

```text
| #  | not           | tcharge | bms_off | not    | sat | ib_choice | ib_amp_fa | ib_noa_fa | cc_diff_fa | ib_dscn_fa | ib_diff_fa | ib_choice | Tb_fa | vb_sel_stat || time_long | accy | off | SAT | flt ekf | fail vb | fail ib | red loss | diff ib | conn |
|    | HDWE_IB_HI_LO | < 24    |         | mod_ib |     | != 0      |           |           |            |            |            |           |       |             ||           |      |     |     |         |         |         |          |         |      |
| -- | ------------- | ------- | ------- | ------ | --- | --------- | --------- | --------- | ---------- | ---------- | ---------- | --------- | ----- | ----------- || --------- | ---- | --- | --- | ------- | ------- | ------- | -------- | ------- | ---- |
| 1  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | T     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 2  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | 0           || ·         | ·    | ·   | ·   | ·       | T       | ·       | ·        | ·       | ·    |
| 3  | ·             | ·       | T       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | T   | ·   | ·       | ·       | ·       | ·        | ·       | ·    |
| 4  | ·             | ·       | ·       | ·      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 5  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | 1         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 6  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | -1        | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 7  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | T          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | T       | ·    |
| 8  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | !=0       | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | T        | ·       | ·    |
| 9  | ·             | ·       | ·       | T      | ·   | ·         | ·         | ·         | ·          | T          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | ·       | T    |
| 10 | T             | ·       | ·       | T      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 11 | T             | ·       | ·       | T      | ·   | ·         | ·         | ·         | ·          | T          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | ·       | T    |
| 12 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | T          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | T       | ·    |
| 13 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | T        | ·       | ·    |
| 14 | T             | ·       | ·       | T      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 15 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | ·       | T    |
| 16 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | T          | ·          | F          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | T       | ·       | ·       | ·        | ·       | ·    |
| 17 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | T          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | T       | ·       | ·       | ·        | ·       | ·    |
| 18 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | ·       | ·    |
| 19 | ·             | T       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || T         | ·    | ·   | ·   | ·       | ·       | ·       | ·        | ·       | ·    |
| 20 | ·             | ·       | ·       | ·      | T   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | T   | ·       | ·       | ·       | ·        | ·       | ·    |
| 21 | ·             | ·       | ·       | ·      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | ·    | ·   | ·   | ·       | ·       | T       | ·        | ·       | ·    |
| 22 | ·             | ·       | ·       | ·      | ·   | T         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·         | T    | ·   | ·   | ·       | ·       | ·       | ·        | ·       | ·    |
```


---

## Annunciation - Serial

### serial_display()

```text
| #  | not           | tcharge | bms_off | not    | sat | ib_choice | ib_amp_fa | ib_noa_fa | cc_diff_fa | ib_dscn_fa | ib_diff_fa | ib_choice | Tb_fa | vb_sel_stat || Tb character | Vb character | Ib character | AH ekf                | time character | AH cc                 |
|    | HDWE_IB_HI_LO | < 24    |         | mod_ib |     | != 0      |           |           |            |            |            |           |       |             || flash        | flash        | flash        | character flash       | flash          | character flash       |
| -- | ------------- | ------- | ------- | ------ | --- | --------- | --------- | --------- | ---------- | ---------- | ---------- | --------- | ----- | ----------- || ------------ | ------------ | ------------ | --------------------- | -------------- | --------------------- |
| 1  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | T     | ·           || ****         | ·            | ·            | ·                     | ·              | ·                     |
| 2  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || Tb           | ·            | ·            | ·                     | ·              | ·                     |
| 3  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | 0           || ·            | *fail        | ·            | ·                     | ·              | ·                     |
| 4  | ·             | ·       | T       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | off          | ·            | ·                     | ·              | ·                     |
| 5  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | Voc          | ·            | ·                     | ·              | ·                     |
| 6  | ·             | ·       | ·       | ·      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | *fail        | ·                     | ·              | ·                     |
| 7  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | 1         | ·     | ·           || ·            | ·            | *fail        | ·                     | ·              | ·                     |
| 8  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | -1        | ·     | ·           || ·            | ·            | *amp         | ·                     | ·              | ·                     |
| 9  | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | T          | ·         | ·     | ·           || ·            | ·            | *diff        | ·                     | ·              | ·                     |
| 10 | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | !=0       | ·     | ·           || ·            | ·            | redl         | ·                     | ·              | ·                     |
| 11 | ·             | ·       | ·       | T      | ·   | ·         | ·         | ·         | ·          | T          | ·          | ·         | ·     | ·           || ·            | ·            | conn         | ·                     | ·              | ·                     |
| 12 | T             | ·       | ·       | T      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | *fail        | ·                     | ·              | ·                     |
| 13 | T             | ·       | ·       | T      | ·   | ·         | ·         | ·         | ·          | T          | ·          | ·         | ·     | ·           || ·            | ·            | conn         | ·                     | ·              | ·                     |
| 14 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | T          | ·         | ·     | ·           || ·            | ·            | * diff       | ·                     | ·              | ·                     |
| 15 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | redl         | ·                     | ·              | ·                     |
| 16 | T             | ·       | ·       | T      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | *fail        | ·                     | ·              | ·                     |
| 17 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | conn         | ·                     | ·              | ·                     |
| 18 | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | Ib           | ·                     | ·              | ·                     |
| 19 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | T          | ·          | F          | ·         | ·     | ·           || ·            | ·            | ·            | ----                  | ·              | ·                     |
| 20 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | T          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ----                  | ·              | ·                     |
| 21 | T             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | Amp_hrs_remaining_ekf | ·              | ·                     |
| 22 | ·             | T       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ·                     | ----           | ·                     |
| 23 | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ·                     | tcharge        | ·                     |
| 24 | ·             | ·       | ·       | ·      | T   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ·                     | ·              | SAT                   |
| 25 | ·             | ·       | ·       | ·      | ·   | ·         | T         | T         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ·                     | ·              | fail                  |
| 26 | ·             | ·       | ·       | ·      | ·   | T         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ·                     | ·              | accy                  |
| 27 | ·             | ·       | ·       | ·      | ·   | ·         | ·         | ·         | ·          | ·          | ·          | ·         | ·     | ·           || ·            | ·            | ·            | ·                     | ·              | Amp_hrs_remaining_soc |
```
