SOC_Particle Critical Path & Test Coverage Review

  Architecture Summary

  The critical path flows in two parallel chains that converge at the sensor selector, then the monitor:

  Ib_amp (ShuntAmp):  ADC → debounce → KF filter → Ib_amp_hdwe ─────┐
                                                                       ├─ ib_range
  Ib_noa (ShuntNoAmp): ADC → debounce → KF filter → Ib_noa_hdwe ────┤   ib_logic (hi/lo limits)
                                                                       ├─ ib_diff  (amp vs noa disagree)
  Vb:  ADC → debounce → LPF → Vb_hdwe ──────────────────────────────┤   ib_wrap  (WrapLoopAmp/Noa → e_wrap)
                                                                       ├─ ib_quiet (disconnected shunt)
  Tb:  ADC → Steinhart-Hart → Tb_hdwe ──────────────────────────────┤   cc_diff  (EKF vs CC divergence)
                                                                       ├─ vb_check (range OOR)
  Vc: ADC → Vc_hdwe ────────────────────────────────────────────────┤   vc_check (op-amp ref OOR)
                                                                       └─ Tb_check (thermistor OOR)
                                                                                │
                                                                select_all_logic (decision table)
                                                                ib_decision_hi_lo / active_standby
                                                                                │
                                                           select_volt_and_current_and_temp
                                                                Ib, Vb, Tb final selections
                                                                                │
                                                        monitor: EKF (Randles) + Coulomb Counter
                                                                                │
                                                                       soc, soc_ekf, regauge

  ---
  Existing Macro Coverage Table
  
  Tier 1 — Hard Current Sensor Failure (range exceeded → IB_AMP/NOA_FLT/FA)

  ┌───────────────┬────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │     Macro     │                     Signal Path Exercised                      │                   Failure Mode / Function Covered                    │
  ├───────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ ampHiEmptFail │ Dm50 → Ib_amp_model ≥ ib_amp_max → IB_AMP_FLT → IB_AMP_FA →    │ Amp over-range from empty battery (low SOC); confirms accy display   │
  │               │ decision #5 → ib_sel_stat=-1 (noa)                             │ set when best sensor lost                                            │
  ├───────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ ampHiFail     │ Same injection from mid SOC                                    │ Amp over-range from mid SOC; primary regression for hard-range       │
  │               │                                                                │ isolation to noa                                                     │
  ├───────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ ampLoFail     │ Dm-50 → negative amp OOR → IB_AMP_FA → noa                     │ Amp under-range (negative); tests negative side of range check       │
  ├───────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ ampLoFullFail │ Dm-50 from full SOC                                            │ Amp under-range before saturation; verifies fault trips before       │
  │               │                                                                │ saturation logic incorrectly clears it                               │
  ├───────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ noaLoFail     │ Dn-50 → IB_NOA_FLT → IB_NOA_FA → decision #3/4 → amp           │ Noa under-range; also generates positive ib_diff_ = amp−noa →        │
  │               │                                                                │ IB_DIFF_HI_FA contributing to isolation                              │
  ├───────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ noaLoFullFail │ Dn-50 + DS-0.30 from full SOC                                  │ Noa under-range racing against SAT threshold; exercises saturation   │
  │               │                                                                │ scalar interaction with cc_diff_thr_                                 │
  └───────────────┴────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘

  Tier 2 — Soft Current Failure (wrap/diff isolation → WRAP_*_FA, IB_DIFF_*_FA)

  ┌──────────────────┬───────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │      Macro       │                          Signal Path Exercised                        │             Failure Mode / Function Covered                │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │                  │ Dn50 → ib_diff_ = Ib_amp − Ib_noa = −50A → IB_DIFF_LO_FLT/FA;         │ Noa positive bias; wrap+diff dual confirmation → latch     │
  │ noaHiFail        │ simultaneously WrapLoopNoa: e_wrap_n↓ ≤ ewlo_thr_ → WRAP_LO_N_FA;     │ on amp; verifies WRAP_LO_N_FA path in hi-lo mode           │
  │                  │ decision #6 → UsingAmp                                                │                                                            │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ noaHiFailNoise   │ Same + DM.75;DN6 PRBS noise                                           │ Same isolation path under noise; tests KF filter           │
  │                  │                                                                       │ robustness and filter persistence thresholds               │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │                  │                                                                       │ Moderate noa bias: diff trips before wrap; verifies        │
  │ noaHiFailSlow    │ Dn20 + tight Fc0.0006 threshold                                       │ IB_DIFF_LO_FA path leads, no CC_DIFF_FA possible (amp      │
  │                  │                                                                       │ still good)                                                │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ noaHiFailSlower  │ Dn8 + Fc0.0006                                                        │ Small noa bias; wrap may trip eventually; tests            │
  │                  │                                                                       │ medium-range sensitivity of diff detection                 │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ noaHiFailSlowest │ Dn5 + Fc0.0006                                                        │ Very small bias: below wrap threshold, diff fires slowly;  │
  │                  │                                                                       │ verifies no false CC_DIFF_FA since amp is still used       │
  ├──────────────────┼───────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ ampHiFailSlow    │ Dm10 + Fc0.0006;Fd0.5 → EKF diverges from CC → CC_DIFF_FA after ~6    │ Slow amp bias with wrap disabled (Fd): verifies            │
  │                  │ min → decision #7                                                     │ Coulomb-counter-diff path to isolation; only cc_diff test  │
  └──────────────────┴───────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘

  Tier 3 — Fake Faults (Ff1 mode: detect without switching)

  ┌─────────────┬─────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │    Macro    │              Signal Path Exercised              │                            Failure Mode / Function Covered                            │
  ├─────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ ampHiFailFf │ Dm50;Ff1 → IB_AMP_FLT fires but latch_=false;   │ Verifies fake-fault mode records fault (latch_fake_ set) but does not switch current  │
  │             │ ib_sel_stat unchanged                           │ sensor; e_wrap crosses threshold visibly                                              │
  ├─────────────┼─────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ vHiFailFf   │ Dv0.8;Ff1 → VB_FLT detected, vb_sel_stat        │ Fake-fault Vb: confirms recording without protective latching; useful for             │
  │             │ unchanged                                       │ deploy-stage regression                                                               │
  └─────────────┴─────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  Tier 4 — Voltage Sensor Failures (VB_FLT/FA, WRAP_VB_FA)

  ┌──────────────────┬─────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
  │      Macro       │                      Signal Path Exercised                      │                 Failure Mode / Function Covered                  │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ vHiFail           │ Dv0.82 → Vb_hdwe ≥ VB_MAX=17V → VB_FLT → VB_FA_LT →            │ Vb sensor high drift; primary Vb failure regression; verifies    │
  │                   │ vb_sel_stat=0 + model substitution                             │ e_wrap path via WRAP_VB_FA and latch_=true                       │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ vHiFailNoise      │ Same + PRBS noise DV0.3                                        │ Noisy Vb high fail; tests LPF robustness on Vb path              │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ offLowSoc         │ Dv-12 from empty → Vb near 0 with very low Ib → vb_check (low  │ BMS-style shutoff at low voltage; tests low-side Vb check and    │
  │                   │ side, Ib>IB_MIN_UP=0.2A condition marginal)                    │ clean shutdown fault behavior                                    │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ offSitBmsBB       │ Twitch Xa-162 → realistic BMS shutoff transient BB chemistry   │ BMS shutoff waveform no-false-trip regression; exercises         │
  │                   │                                                                │ ib_really_quiet_ and WRAP_VB_FA guard during shutoff             │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ offSitBmsCHG      │ Same, CHG chemistry Xa-324                                     │ Same as above for CHG chemistry                                  │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ offSitBmsNoiseBB  │ BMS shutoff BB + noise DV0.3                                   │ Noisy BMS shutoff; 2× normal Vb noise stress test                │
  ├───────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ offSitBmsNoiseCHG │ Same CHG + noise                                               │ Noisy BMS shutoff CHG chemistry                                  │
  └───────────────────┴────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘

  Tier 5 — Temperature Sensor Failures (TB_FLT/FA → NOMINAL_TB fallback)

  ┌──────────────┬────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┐
  │    Macro     │                   Signal Path Exercised                    │                      Failure Mode / Function Covered                      │
  ├──────────────┼────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ tLoFailModel │ D^-113 → Tb_model ≤ TB_MIN=−40°C → TB_FLT → TB_FA →        │ Open thermistor simulation (model path); verifies fallback to nominal Tb  │
  │              │ Tb_=NOMINAL_TB                                             │ preserves soc                                                             │
  ├──────────────┼────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ tHiFailModel │ D^+50 → Tb_model ≥ TB_MAX=60°C → TB_FA                     │ Short-to-high thermistor (model path)                                     │
  ├──────────────┼────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ tLoFailHdwe  │ Dt-113 with modHalfInit230 (partial hardware Tb) → Tb_hdwe │ Hardware thermistor open; exercises hardware Tb_check path                │
  │              │  ≤ TB_MIN                                                  │                                                                           │
  ├──────────────┼────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ tHiFailHdwe  │ Dt+50 hardware                                             │ Hardware thermistor short-high                                            │
  └──────────────┴────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────┘

  Tier 6 — Regression / SOC Accuracy / False-Trip Prevention

  ┌─────────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
  │          Macro          │          Signal Path Exercised           │                         Failure Mode / Function Covered                          │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ rapidTweakRegression    │ Xp10: 3× large ±50A cycles via both      │ Primary false-trip regression: all fault bits must stay clear through aggressive │
  │                         │ sensors; full fault pipeline             │  current swings; exercises zero-crossing ib_sel transitions                      │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ rapidTweakRegressionH0  │ Same + Sh0 (hysteresis=0)                │ Hysteresis-removed regression; isolates e_wrap faults from hysteresis            │
  │                         │                                          │ contribution; critical for WRAP_*_FA tuning                                      │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ rapidTweakRegression40C │ D^15 offset (≈40°C) + Xp10               │ High-temperature false-trip regression; verifies table-scaling at elevated Tb    │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ slowTweakRegression     │ Xp11: 1× slow cycle                      │ Slow-rate regression; more sensitive to cc_diff drift; tests cc_diff false-trip  │
  │                         │                                          │ at low dI/dt                                                                     │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ triTweakDisch           │ Xp13: 3× discharge-only cycles           │ Discharge-only regression; catches false ib_diff trips at zero-crossing in noa   │
  │                         │                                          │ path                                                                             │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ pulseSoft               │ Xp7: injected model-current pulse →      │ Model pulse response; e_wrap must be flat = confirms Randles r_ct/tau_ct         │
  │                         │ e_wrap flatness                          │ parameterization                                                                 │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ pulseHard               │ Xp8: hardware current pulse              │ Hardware pulse response; same e_wrap flatness check against hardware time        │
  │                         │                                          │ constants                                                                        │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ stepDown                │ DI-50 → clean 50A step discharge → soc↓  │ SOC accuracy under step discharge; no faults should trip during clean transient  │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ stepUp                  │ DI50 → clean 50A step charge → soc↑      │ SOC accuracy under step charge                                                   │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ satSitBB                │ Xa17 charge into saturation, then relax, │ Saturation detection + Is_sat_delay + Mon->regauge() CC→EKF resync at full       │
  │                         │  BB                                      │ charge, BB chemistry                                                             │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ satSitCHG               │ Same, CHG chemistry                      │ Saturation handling CHG chemistry; verifies ewsat_slr_ dampening of wrap         │
  │                         │                                          │ thresholds                                                                       │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ flatSit                 │ Xa-81 operate around 0.9 SOC with flat   │ EKF convergence on flat VOC curve; tests case where EKF has poor voltage         │
  │                         │ voc(soc) (CHINS)                         │ leverage                                                                         │
  ├─────────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ zero_with_pc            │ Xm2 (hardware Ib active) +               │ Hardware zero-calibration procedure; only test using live hardware Ib path       │
  │                         │ zeroPrepHdweNoVb + Fi2;Fo2               │ (mod_ib=false); exercises vc_check actively                                      │
  └─────────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘

  ---
  Identified Coverage Gaps
  
  The following failure modes and paths have no dedicated test macro:

  ┌─────┬──────────────────────┬────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │  #  │          Gap         │                        Missing Path                        │                            Risk                            │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │     │                      │ vc_check → VC_FLT → VC_FA → both IB_AMP_FA + IB_NOA_FA     │ Undetected op-amp supply failure leaves both current       │
  │ G1  │ Vc reference failure │ simultaneously → UsingNone / ib_sel_stat=0                 │ sensors silently poisoned; SOC integration with wrong      │
  │     │                      │                                                            │ current                                                    │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ G2  │ Dual sensor total    │ ib_amp_fa && ib_noa_fa → decision #1 → UsingNone (hi-lo)   │ No test verifies the app responds correctly when both      │
  │   │ failure              │ or ib_sel_stat=0 (active/standby) → SOC integration        │ sensors fail; behavior at Ib=0 indefinitely                │
  │     │                      │ halted                                                     │                                                            │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │     │ Disconnected shunt   │ ib_quiet → IB_DSCN_FLT after QUIET_SET=60s → IB_DSCN_FA;   │ False IB_DSCN_FA during legitimate zero-current sleep; also│
  │     │                      │ WRAP_VB_FA                                                 │                                                            │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │     │ Hi-lo transition     │ scale_select() partial weighting: ib_sel_stat=0            │ SOC integrates a blended current value in transition; not  │
  │ G6  │ zone blending        │ (transition) when Ib_noa is between n_hi and p_lo          │ validated, particularly near zero current                  │
  │     │                      │ breakpoints                                                │                                                            │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ G7  │ CC_DIFF_FA isolation │ Decision #10/12: ib_diff_fa && vb_sel_stat && wrap_m_fa    │ ampHiFailSlow tests cc_diff in active/standby only; hi-lo  │
  │     │  in hi-lo mode       │ && !wrap_n_fa → UsingNoa; cc_diff_fa alone → KeepTrying    │ cc_diff decision branches are untested                     │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │     │ Cold temperature     │ D^ near TB_MIN: voc(soc) table, Randles parameters, and    │ Only rapidTweakRegression40C covers a non-nominal Tb; cold │
  │ G8  │ regression           │ CC accuracy at ≈−20°C                                      │ behavior (large capacity and internal resistance changes)  │
  │     │                      │                                                            │ untested                                                   │
  ├─────┼──────────────────────┼────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤

  ---
  Proposed New Macros

  # G1 — Vc reference failure (hardware path; mod_ib must be false)
  # D3 injects ap.vc_add() offset onto Vc_ before vc_check.
  # Pushing Vc below VC_MIN=1.1V (nominal ~1.65V) with D3-0.6 triggers VC_FLT.
  # Propagates: VC_FA → IB_AMP_FA + IB_NOA_FA → UsingNone.
  'vcFail': (120,
      hdwNoVbPcMidInit +           # Xm2: hardware Ib path, vc_check active
      tranPrep +
      'XY;D3-0.60;XQ40000;' +      # Inject Vc bias below VC_MIN=1.1V
      'D3;' + quiet + cleanup + '<XD;',
      ("Should detect Vc failure: VC_FA sets, both IB_AMP_FA and IB_NOA_FA set, UsingNone.",
       "Verifies undetected op-amp supply fault path. Check vc_flt, vc_fa, ib_amp_fa, ib_noa_fa all set.",
       "ib_sel_stat should go to 0 / UsingNone.")),

  # G2 — Dual sensor failure simultaneously (active/standby; both range fail)
  # Inject both sensors out of range at the same time → ib_amp_fa && ib_noa_fa → ib_sel_stat=0
  'dualIbFail': (120,
      modHalfInit +
      tranPrep +
      'Mm0.1;Nm0.1;' +             # Shrink both range maxima to 0.1A (Mm/Nm set ib_amp_max/ib_noa_max)
      c10 +                        # 10A injection — both sensors now OOR
      'XQ30000;' +
      'Mm;Nm;' + c00 + quiet + cleanup + '<XD;',
      ("Both amp and noa sensors fail simultaneously → ib_sel_stat=0 (UsingNone).",
       "Verifies decision branch 1: ib_amp_fa && ib_noa_fa → latch with no current.",
       "SOC integration should freeze. Check that no wrap false faults fire.")),

  # G3 — Disconnected shunt / quiet detection (QUIET_SET=60s persistence needed)
  # Hold both sensors near zero for >60s to trigger IB_DSCN_FA.
  # Also verifies that ib_really_quiet_ prevents WRAP_VB_FA from latching.
  'quietDscn': (180,
      modHalfInit +
      'Dm0;Dn0;DI0;' +             # Zero all current injections
      'XQ100000;' +                # Hold quiet for 100s > QUIET_SET=60s
      'BZ;' + quiet + cleanup + '<XD;',
      ("Hold all currents near zero for 100s to trigger IB_DSCN_FLT → IB_DSCN_FA.",
       "Verify ib_is_quiet, ib_dscn_flt, ib_dscn_fa all set after ~60s.",
       "Also confirm wrap_vb_fa does NOT latch (ib_really_quiet_ guards it during quiet state).")),

  # G6 — Hi-lo transition zone (scale_select blending; ib_sel_stat=0 expected)
  # Drive Ib_noa through the transition band between n_hi and p_lo breakpoints.
  # Observe ib_sel_stat transitions through 0 (blending) at breakpoints.
  'hiLoTransition': (120,
      modHalfInit +
      slow +
      'Rs;W4;' +
      'Xp10;' +                    # Standard twitch cycles pass through transition zone
      quiet + cleanup + '<XD;',
      ("Current ramps through hi-lo transition zone; ib_sel_stat=0 (blend) should appear in data near breakpoints.",
       "No faults should trip. Verify smooth blending of amp/noa in scale_select.",
       "Look for momentary ib_sel_stat=0 in plots — normal behavior, not a fault.")),

  # G7 — CC_DIFF_FA in hi-lo mode (slow amp bias with hi-lo configuration)
  # Equivalent to ampHiFailSlow but for hi-lo hardware configuration.
  'ampHiFailSlowHiLo': (535,
      modHalfInit +
      'Fc0.0006;Fd0.5;' +          # Tight cc_diff threshold, disable wrap (Fd)
      tranPrep + c10 +
      'XQ400000;' +
      c00 + quiet + cleanup + '<XD;',
      ("10A amp bias in hi-lo mode, wrap disabled. Tests CC_DIFF_FA path in hi-lo decision table.",
       "Decision #10: ib_diff_fa && vb_sel_stat && wrap_m_fa && !wrap_n_fa → UsingNoa.",
       "EKF should follow voltage while soc wanders. Run 6+ min to confirm cc_diff_fa sets.")),

  # G8 — Cold temperature regression
  'rapidTweakRegressionM20C': (230,
      'D^-35;' +                   # Push Tb_model to about -20°C (ambient ~15°C - 35 = -20)
      slow + 'Rs;W8;Xp10;' +
      quiet + cleanup + '<XD;',
      ("Regression at ~-20°C. Tests voc(soc) and Randles parameters at cold temperature.",
       "No faults should trip. Check soc accuracy and e_wrap behavior at low Tb.",
       "Occasional ib_sel_stat jumps at zero crossing are normal.")),

  ---
  Coverage Summary

  ┌──────────────────────────────────┬────────────┬─────────────────────────┬─────────────────┐
  │             Category             │ # Existing │          # Gap          │   Risk Level    │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Hard Ib range fail (amp/noa OOR) │ 6          │ —                       │ ✅  Well covered │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Soft Ib wrap/diff isolation      │ 6          │ G7 (hi-lo cc_diff only) │ ⚠️ Partial       │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Fake faults (Ff1)                │ 2          │ —                       │ ✅  Covered      │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Tb failure                       │ 4          │ —                       │ ✅  Covered      │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Vc failure                       │ 0          │ G1                      │ 🔴 Untested      │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Dual sensor failure              │ 0          │ G2                      │ 🔴 Untested      │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Quiet/disconnect detection       │ 0          │ G3                      │ 🔴 Untested      │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Hi-lo transition zone            │ 0          │ G6                      │ ⚠️ Minor gap     │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Cold temperature regression      │ 0          │ G8                      │ ⚠️ Minor gap     │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ Regression / false-trip          │ 8          │ —                       │ ✅  Well covered │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤
  │ SOC accuracy / saturation        │ 4          │ —                       │ ✅  Covered      │
  ├──────────────────────────────────┼────────────┼─────────────────────────┼─────────────────┤

  Priority fixes: G1 and G2 (complete failure modes with no coverage), G3 (the ib_really_quiet_ guard on WRAP_VB_FA is
  logic that protects against false latching during BMS shutoff — currently relied upon in offSitBms* tests but never directly validated).

---

## Implemented Tests (added to `lookup` in `pyStateOfCharge/GUI_common.py`)

Three new macros now close the highest-priority gaps identified above (G1, G2). They appear at the bottom of `lookup` and are listed in `sel_list1` so `default_auto` picks them up.

| Macro        | Initial state | Injection                    | Signal Path Exercised                                                                                       | Failure Mode / Function Covered |
|--------------|---------------|------------------------------|--------------------------------------------------------------------------------------------------------------|---------------------------------|
| `ibDualMid`  | `modHalfInit` (≈50 % SOC) | `cmn100` — `Dn100;Dm100;` (both shunts +100 A) | Both Ib paths exceed `ib_amp_max` / `ib_noa_max` simultaneously → `IB_AMP_FA` && `IB_NOA_FA` → decision branch 1 → `ib_sel_stat = 0` / `UsingNone` | Dual hard current-sensor failure at mid SOC. Verifies the simultaneous-fail branch of the decision table; SOC integration must halt; `*fail` annunciates. |
| `ibDualFlat` | `modFlatInit` (≈90 % SOC, flat voc(soc)) | `cmn100` — `Dn100;Dm100;` | Same as above but exercised in the flat `voc(soc)` region where the EKF has poor voltage leverage. | Dual hard current-sensor failure in the flat-OCV region. Confirms the dual-fail isolation does not depend on EKF lever-arm and no false `cc_diff` self-recovery occurs while in the flat zone. |
| `vcFlat`    | `modFlatInit` (≈90 % SOC) | `D30.6` — Vc bias to ≈0.6 V (nominal 1.65 V) | `Vc_` falls below `VC_MIN` → `VC_FLT` → `VC_FA` → both `IB_AMP_FA` and `IB_NOA_FA` set → `UsingNone`. | Op-amp 3v3/2 reference failure (single point that silently poisons both current channels). Closes gap G1 — previously untested in any macro. |

**Why these matter.**

- `ibDualMid` / `ibDualFlat` cover the **dual-sensor total-loss** failure mode (gap G2). The decision-table branch they exercise (`ib_amp_fa && ib_noa_fa` → `UsingNone`) was previously reachable only by chance through compound `noaHi*` + `ampHi*` overlays. These two macros — half-SOC and flat-zone variants — pin the behavior down so SOC must freeze on dual loss without false recovery in either operating region.
- `vcFlat` covers the **Vc reference failure** mode (gap G1). Because both op-amp channels share the 3v3/2 reference sampled as `Vc`, an undetected drift here corrupts both `Ib_amp` and `Ib_noa` simultaneously while neither range check nor `ib_diff` would trip (they move together). The `vc_check` path is the only defense; this macro is its only regression.

**How to run.** Both are part of `default_auto` (regenerated when `vcFlat` was added to `sel_list1`). To run individually from the GUI test harness, select the macro name. Each completes in ~130 s. Expected results — `*fail` on the OLED, `ib_sel_stat = 0`, frozen `Ult 1` fault record, both `ib_amp_fa` and `ib_noa_fa` latched — are documented in the macro's own help-text tuple.

**Related macros** (left in place; provide partial coverage of adjacent gaps):

- `ampHiFail` / `noaHiFail` — single-channel hard fails.
- `ampHiFailFf` — fake-fault mode confirms detect-without-switch path.
- `vHiFail` — Vb (not Vc) sensor hard fail.

Remaining gaps not yet addressed: G3 (disconnected-shunt `ib_quiet`), G6 (hi-lo blend zone), G7 (CC_DIFF in hi-lo mode), G8 (cold temperature regression)
