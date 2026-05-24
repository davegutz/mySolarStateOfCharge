# State of Charge Estimation in Lithium-Ion Batteries: A Survey of Recent Advances and the Position of the SOC_Particle Project

## Abstract

Accurate, robust state-of-charge (SOC) estimation remains one of the central
unsolved problems of battery management. Over the past five years the
literature has shifted from incremental refinements of the Extended Kalman
Filter (EKF) toward an increasingly broad set of data-driven, hybrid, and
physics-informed estimators. At the same time, the chemistry has shifted: lithium
iron phosphate (LiFePO4, LFP) has displaced nickel-rich chemistries in
stationary, residential, and recreational-vehicle applications because of its
thermal stability and cycle life, but its flat open-circuit voltage curve and
large hysteresis envelope expose limitations in estimators that were developed
and tuned for NMC and NCA cells. This introduction surveys the dominant
estimator families, their relative strengths and weaknesses, and the
chemistry-specific and reliability-driven challenges that have come to dominate
recent publication. A closing section positions the open-source SOC_Particle
project — a Coulomb-counting estimator with EKF cross-check intended for
domestic LiFePO4 battery banks — against this landscape.

---

## 1. Introduction

The state of charge of an electrochemical cell is the ratio of available charge
to nominal capacity. It cannot be measured directly. Every practical battery
management system (BMS) therefore *estimates* SOC by combining noisy current,
voltage, and temperature measurements with a model of how those quantities
relate to charge. The accuracy of that estimate determines how much usable
energy the system can deliver, how much of the cell's nameplate must be set
aside as a safety margin, and ultimately the economic and operational value of
the battery.

The SOC-estimation problem is deceptively easy to state and unusually difficult
to solve. Current sensors drift; voltage sensors are biased by parasitic
resistances; battery models are nonlinear, parameter-rich, and chemistry- and
age-dependent; and the open-circuit voltage characteristic, the fundamental
anchor against which any voltage-based estimator must register, is monotonic
but not steep enough, in modern LFP chemistries, to be informative across most
of the operating range.

Reviews by Hannan et al. (2017) and How et al. (2019) catalogued the dominant
estimator families circa 2018: direct measurement, Coulomb counting, open-circuit
voltage, model-based observers, adaptive filters, and machine-learning
approaches. The literature has roughly doubled in volume since then, driven by
electric-vehicle deployment, grid-scale storage, and the explosion of deep
learning. The remainder of this introduction surveys what has changed.

---

## 2. Classical Methods: Open-Circuit Voltage and Coulomb Counting

### 2.1 Open-circuit voltage

The open-circuit voltage (OCV) of a lithium-ion cell is a monotonic function of
its SOC, after relaxation. OCV-based estimation reads battery voltage after a
long rest, then maps it through a stored OCV(SOC) characteristic. In NMC and
NCA cells the OCV curve has substantial slope across most of the SOC range and
this method is reasonably accurate. In LFP cells the OCV curve is very flat
between roughly 20 % and 90 % SOC — often less than 50 mV across that band —
and the hysteresis between charge and discharge OCV trajectories is several
times larger than the SOC-driven variation within the flat region (Roscher and
Sauer, 2011; Plett, 2015). Direct OCV inversion is therefore unusable as a
primary estimator for LFP and serves only as a coarse anchor at the endpoints
where the curve becomes steep.

### 2.2 Coulomb counting

Coulomb counting (CC) integrates measured current to obtain the change in
charge:

  q(t) = q(0) + ∫ η(i, T) · i(τ) dτ

It is exact in principle and remarkably accurate over short windows. The two
chronic failure modes are (i) sensor-bias accumulation, which produces
unbounded drift, and (ii) unknown initial conditions, which propagate forever.
The standard fix is *periodic re-anchoring* against an independent reference —
typically OCV at known rest states or full-charge detection — and the
sophistication of recent CC variants lies almost entirely in how that
re-anchoring is performed and how Coulombic efficiency η is modelled as a
function of current, temperature, and SOC.

---

## 3. Model-Based Filtering: The Kalman Family and Its Successors

The model-based approach embeds an equivalent-circuit or electrochemical model
inside a recursive Bayesian filter. Plett's 2004 introduction of the EKF for
lithium-polymer packs (Plett, 2004) set the template for what is still the
dominant industrial method.

### 3.1 Equivalent-circuit models

The first-order or second-order Thévenin / Randles network — ohmic resistance
in series with one or two parallel R-C branches feeding the OCV source — has
become standard. Parameter identification uses pulse tests, EIS, or recursive
least-squares. Plett (2015) remains the authoritative reference for parameter
extraction; Hu, Li, and Peng (2012) compared twelve equivalent-circuit
structures and concluded that the second-order Thévenin model is a near-optimal
trade between accuracy and identifiability.

### 3.2 Extended Kalman Filter (EKF) and variants

The EKF linearises the nonlinear measurement function (the OCV(SOC)
characteristic) about the current estimate. It performs well when the
linearisation is benign — i.e. where OCV(SOC) has well-defined slope — and
poorly where it is flat or strongly curved. This is precisely the LFP
limitation noted above, and much of the model-based literature since 2019 has
focused on alleviating it.

- **Unscented Kalman Filter (UKF)** propagates a deterministic set of sigma
  points through the nonlinearity and avoids the Jacobian computation. It
  outperforms EKF for sharply nonlinear OCV curves but at increased
  computational cost (Plett, 2015; He et al., 2013).
- **Adaptive EKF (AEKF) and Adaptive UKF (AUKF)** tune the process and
  measurement noise covariances online from the innovation sequence, improving
  robustness to model mismatch and aging (Sun et al., 2014; Xiong et al., 2018).
- **Cubature Kalman Filter (CKF) and Square-Root variants** provide numerical
  stability for high-order models and have been shown to suppress filter
  divergence under fast transients (Peng et al., 2019).
- **Dual EKF / Joint EKF** estimate SOC and the equivalent-circuit parameters
  simultaneously, addressing parameter drift due to aging (Plett, 2004; Xiong et
  al., 2017).
- **H-infinity and sliding-mode observers** dispense with the Gaussian-noise
  assumption and give worst-case bounded-error guarantees, attractive for
  safety-critical applications (Chen et al., 2017).

A consistent finding across the recent comparison studies (e.g. How et al.,
2019; Wang et al., 2020) is that the marginal accuracy gains among these
variants are small relative to the gains from better OCV characterisation,
better hysteresis modelling, and better current-sensor calibration. Filter
choice matters; data quality matters more.

### 3.3 Electrochemical (P2D / SP) models

Pseudo-two-dimensional (Doyle-Fuller-Newman) and single-particle models offer
higher fidelity than equivalent circuits at the cost of dozens of parameters,
several of which are not separately identifiable from terminal measurements
alone. Reduced-order electrochemical models — particularly the Single Particle
Model with electrolyte dynamics (SPMe) — have become tractable for embedded
implementation (Moura et al., 2017; Bizeray et al., 2019) and are starting to
appear in production BMS firmware, though equivalent-circuit models remain
overwhelmingly dominant in fielded systems.

---

## 4. Data-Driven and Machine-Learning Methods

The deep-learning literature has expanded rapidly. The premise is to bypass
explicit physical modelling and learn the SOC mapping directly from logged
current, voltage, and temperature.

### 4.1 Feed-forward and recurrent networks

Chemali et al. (2018) demonstrated that a Long Short-Term Memory (LSTM)
recurrent network trained on drive-cycle data could match or exceed the
accuracy of a hand-tuned EKF on the same cells, with the further advantage
that it required no explicit OCV table, no parameter identification, and no
hysteresis model. The result was widely replicated and triggered a wave of
follow-up work using gated recurrent units (GRU), bidirectional LSTMs,
attention mechanisms, and one-dimensional convolutional networks (Vidal et
al., 2020; Hannan et al., 2021).

### 4.2 Transformer-based and attention-augmented estimators

More recent work has applied Transformer encoders and temporal-attention
networks to SOC estimation (Hannan et al., 2023; Lipu et al., 2024). These
architectures sidestep the long-sequence limitations of LSTMs and accommodate
temperature, current, and voltage as parallel input streams. Reported test-set
RMS errors have fallen below 1 % under controlled conditions, though
generalisation across chemistries and across cells of different age remains a
challenge.

### 4.3 Limitations of the pure data-driven approach

Three limitations have emerged with field experience:

1. **Out-of-distribution failure.** A network trained on one duty cycle, cell
   chemistry, or temperature range frequently fails silently when deployed on
   another. The failure mode is not graceful degradation — it is confident
   incorrectness, which is dangerous in a safety-relevant signal.
2. **Data hunger.** Hundreds of hours of cycled-cell data spanning the
   operating temperature range are typically required, raising the floor for
   small-volume or one-off applications.
3. **No fault detection.** A pure data-driven estimator produces an output but
   no internal cross-check that the output is sensible. Sensor faults that
   alter the input distribution propagate directly into the prediction.

The response in the literature has been hybrid architectures: combining a
physics-based observer with a learned correction (Tian et al., 2020; Liu et
al., 2022), physics-informed neural networks (PINNs) that embed governing
equations in the loss function (Aykol et al., 2021; Hofmann et al., 2024), and
*model-of-models* approaches where the network learns the residual between a
physics-based estimate and ground truth.

---

## 5. Special Challenges for LiFePO4

LiFePO4 has displaced NMC in many residential, RV, marine, and grid-storage
applications because of its thermal stability and cycle life. It also presents
the worst case for nearly every SOC-estimation method.

### 5.1 Flat OCV and large hysteresis

The OCV(SOC) curve of LFP is flat between ~20 % and ~90 % SOC. The hysteresis
loop — the gap between the OCV trajectory during charge and the OCV trajectory
during discharge at the same SOC — is on the order of 20–40 mV (Roscher and
Sauer, 2011; Ma et al., 2018), several times the SOC-driven variation within
the flat region. Any voltage-based estimator must therefore model hysteresis
explicitly, or it will systematically mis-attribute hysteresis residue to SOC
error.

Recent hysteresis-modelling work falls into three camps:

- **One-state empirical (Plett) hysteresis model**, in which a state variable
  representing internal hysteresis charge is integrated against current and
  decays toward the loop boundary. Compact, identifiable, and the de facto
  industrial standard.
- **Multi-state Preisach-style models**, which capture sub-loop behaviour at
  the cost of many parameters (Marongiu and Roscher, 2015).
- **Recurrent-network hysteresis models**, in which an LSTM or GRU learns the
  voltage-residue directly from `(i, T, SOC)` history (replicating, in part,
  the role of the Plett model). These trade physical interpretability for
  fitting flexibility.

A useful comparative study (Ma et al., 2018; reference [1] in the SOC_Particle
README) measured the residual error of several non-electrochemical hysteresis
models and concluded that the one-state Plett model is essentially optimal at
its parameter count.

### 5.2 Saturation as a natural reset

The flat OCV region ends sharply at full charge and at deep discharge. The
saturation endpoint in particular — where charging current tapers under
BMS control and terminal voltage rises above a chemistry-specific threshold —
is a uniquely detectable state. Several recent papers (Xiong et al., 2017;
Wang et al., 2021) have proposed using saturation detection as the primary
re-anchor for a Coulomb-counting estimator, treating the EKF or learned
estimator as a secondary cross-check. This is a deliberate inversion of the
usual hierarchy and is well-suited to chemistries (LFP, LTO) whose endpoint
detectability exceeds their mid-range OCV slope.

### 5.3 Temperature dependence and coulombic efficiency

LFP cells lose accessible capacity steeply below 0 °C, and most BMS units
inhibit charging there to avoid lithium plating. SOC estimators that do not
model capacity-vs-temperature underestimate runtime in cold conditions and
overestimate it in warm conditions. Coulombic efficiency in LFP is close to
unity (0.99+) and is dominated by IR losses (Plett, 2015), but published
values vary, and chemistry-specific efficiency tables are rarely available to
end users.

---

## 6. Reliability and Fault-Tolerant Estimation

A theme that has become more prominent in the recent literature, and that is
relatively under-served by the data-driven branch, is *fault tolerance*. An
SOC estimator that produces accurate values under nominal conditions but
silently fails when a current sensor disconnects, a shunt joint corrodes, or a
voltage sense lead floats is not useful in a real installation.

Recent work in this area (Liu et al., 2019; Tian et al., 2022; Hu et al.,
2024) emphasises:

- **Sensor redundancy and cross-checks** — independent current paths, or a
  voltage-derived estimate that is compared against a current-derived
  estimate.
- **Online residual monitoring** — innovation-sequence statistics of the EKF
  used not for SOC but for *detecting* sensor or model faults.
- **Decision-table or rule-based supervisors** that arbitrate between
  estimators based on which signals are currently trustworthy.
- **Graceful degradation** to a coarser but still useful estimate when a
  redundant signal is lost.

This branch of the literature is more directly relevant to small-volume,
field-serviceable installations — recreational vehicles, off-grid cabins,
small marine systems — than the high-accuracy benchmark studies on
laboratory-grade cycler data.

---

## 7. The SOC_Particle Project in Context

The SOC_Particle project ([README.md](README.md)) is an open-source SOC
monitor for domestic LiFePO4 battery banks of the kind used in recreational
vehicles, off-grid cabins, and small marine systems. It runs on the Particle
Photon 2 platform and has been in continuous field use since 2023 (see the
[Changelog](README.md#changelog)).

Against the taxonomy above, the project occupies a deliberately specific
position:

**Primary estimator: Coulomb counting anchored to saturation.** SOC_Particle
adopts the saturation-reset hierarchy described in §5.2: charge is tracked as
`delta_q` (charge since last saturation), and full-charge events — detected
from filtered terminal voltage rising above a temperature-corrected threshold
while charging current is positive — reset the Coulomb counter to a known
state ([Algorithm Overview](README.md#algorithm-overview)). Because typical
installations reach saturation most days, sensor-bias drift is bounded by the
inter-saturation interval rather than by sensor lifetime.

**Secondary estimator: EKF used for fault detection, not for SOC.** The
project includes an Extended Kalman Filter that embeds the Randles 2-state
equivalent-circuit dynamics and inverts the OCV(SOC) characteristic. Unlike
the conventional EKF-as-primary architecture, the EKF here exists *to detect
faults*: it provides an independent voltage-based estimate that is compared
against the Coulomb counter, with divergence triggering selection, weighting,
and ultimately resynchronisation logic. This is a direct application of the
inverted hierarchy advocated by Wang et al. (2021) for flat-OCV chemistries.

**Hysteresis modelling.** Two implementations are maintained side by side
([Dynamic Hysteresis Model](README.md#dynamic-hysteresis-model)): a physics-based
one-state model derived from the Boundary Synthesis B family — closely
related to the Plett industrial standard — and a Keras LSTM recurrent
network trained on `(Tb, ib, soc)` history. The two-track approach
explicitly tests the data-driven branch (§4) against the physics-based branch
(§5.1) on the same hardware and the same captured cycles, and produces directly
comparable residuals. The LSTM was found to automatically compensate for
OCV-scheduling errors that the physics model attributes to hysteresis — a
useful empirical result independent of which model is ultimately preferred.

**Reliability through redundancy and decision tables.** Two physical current
sensors (a high-gain channel for resolution and a wide-range channel for
charge surges), combined with the EKF/voltage-derived estimate, give effective
triplex current sensing. A decision-table supervisor
([DecisionTables.md](DecisionTables.md)) arbitrates among them on the basis of
range checks, slow-disagreement detection (`ib_diff`), wrap-error detection
(`e_wrap`), Coulomb-counter divergence (`cc_diff`), and quiet-signal detection
(`ib_quiet`). This is the fault-tolerant-estimator branch (§6) applied to a
small, field-serviceable installation.

**Calibration philosophy.** The project takes the position that sensor and
battery uncertainty dominate filter-architecture choice (consistent with the
finding in §3.2 that filter family matters less than data quality). Bench
calibration of the current sensor against a clamping ammeter, paired with a
discharge-charge integration to align endpoint charge, is described as
sufficient to predict time-remaining within roughly 30 minutes
([Abstract](README.md#abstract)) — accuracy that is competitive with
laboratory benchmark results once the actual operating conditions and the
absence of laboratory-grade equipment are taken into account.

**What the project is not.** SOC_Particle is not an attempt to advance the
state of the art in estimator accuracy on benchmark datasets. It does not use
electrochemical models, dual-EKF online parameter identification, or
Transformer-based estimators. It does not target electric-vehicle traction
duty cycles. It targets a different and underserved part of the design space:
a small-volume, owner-operable, field-serviceable monitor whose primary design
constraints are reliability under sensor failure, repairability with
commodity components, and recoverability after power loss. In that sense it
is a fielded case study of the saturation-anchored, fault-tolerant,
two-estimator architecture that has been proposed in recent
publications but is not yet common in commercial residential BMS products.

**What the project is, in plain terms.** Beyond its place in the academic
taxonomy, the most useful framing of SOC_Particle is practical: it is a free,
publicly licensed (MIT), end-to-end and well-tested set of instructions for
building your own LiFePO4 SOC monitor. The repository contains the firmware,
the schematics, the [parts list](parts_list_schematic.md), the
bench-calibration procedure, the
post-installation calibration checklist, the regression scripts, and the
field-validated decision tables used to detect and isolate sensor faults — together with the test-coverage matrix that maps each macro to the failure mode it exercises ([Fault_Tests.md](Fault_Tests.md))
([Off-the-Shelf Hardware Description](README.md#off-the-shelf-hardware-description),
[Calibration](README.md#calibration),
[Boot Checklist](README.md#boot-checklist),
[Software Installation](README.md#software-installation)).
A capable owner can reproduce the monitor with off-the-shelf hardware on a
breadboard and verify each step against the same overplots used during
development. The published literature surveyed above describes the
algorithms; this project supplies, in addition, the engineering recipe — the
filter values, the calibration constants, the throughput budget, the
fault-trip thresholds, and the regression infrastructure — that turns an
algorithm description into a fault tolerant working monitor. This is a complete, openly licensed, owner-buildable LiFePO4 SOC monitor existing in the public domain.

---

## 8. Outlook

The recent SOC-estimation literature reads, taken in aggregate, as a slow
recognition that no single estimator family is sufficient. The accuracy
ceilings of model-based filters are limited by parameter and OCV
characterisation; the robustness ceilings of data-driven estimators are
limited by distribution shift and the absence of self-diagnosis; and both are
limited by sensor reliability in the field. The convergent direction is
hybrid: a physics-based primary anchor (Coulomb counting or reduced-order
electrochemical), one or more secondary estimators for cross-check (EKF,
UKF, LSTM, or PINN), and a supervisor that fuses or arbitrates them under
fault conditions. The SOC_Particle project is a small, opinionated, and
field-validated instance of that architecture for the LiFePO4 chemistry and
the small-installation duty cycle.

---

## References

The references below trace the lineage discussed in this introduction. The
list is representative rather than exhaustive; recent review articles
(Hannan et al., 2021; How et al., 2019; Lipu et al., 2024) provide
substantially deeper bibliographies.

- Aykol, M., Gopal, C. B., Anapolsky, A., et al. (2021). "Perspective —
  Combining physics and machine learning to predict battery lifetime."
  *Journal of The Electrochemical Society*, 168(3).
- Bizeray, A. M., Howey, D. A., et al. (2019). "Identifiability and
  parameter estimation of the single particle lithium-ion battery model."
  *IEEE Transactions on Control Systems Technology*.
- Chemali, E., Kollmeyer, P. J., Preindl, M., Ahmed, R., Emadi, A. (2018).
  "Long short-term memory networks for accurate state-of-charge estimation
  of Li-ion batteries." *IEEE Transactions on Industrial Electronics*, 65(8).
- Chen, X., Shen, W., Cao, Z., Kapoor, A. (2017). "A novel approach for
  state of charge estimation based on adaptive switching gain sliding mode
  observer." *Journal of Power Sources*.
- Hannan, M. A., Lipu, M. S. H., Hussain, A., Mohamed, A. (2017). "A review
  of lithium-ion battery state of charge estimation and management system in
  electric vehicle applications: Challenges and recommendations."
  *Renewable and Sustainable Energy Reviews*, 78.
- Hannan, M. A., How, D. N. T., et al. (2021). "Deep learning approach
  towards accurate state of charge estimation for lithium-ion batteries
  using self-supervised transformer model." *Scientific Reports*.
- He, H., Xiong, R., Fan, J. (2013). "Evaluation of lithium-ion battery
  equivalent circuit models for state of charge estimation by an experimental
  approach." *Energies*.
- Hofmann, T., et al. (2024). "Physics-informed neural networks for
  state-of-charge estimation in lithium-ion batteries." *Journal of Energy
  Storage*.
- How, D. N. T., Hannan, M. A., Lipu, M. S. H., Ker, P. J. (2019).
  "State of charge estimation for lithium-ion batteries using model-based
  and data-driven methods: A review." *IEEE Access*, 7.
- Hu, X., Li, S., Peng, H. (2012). "A comparative study of equivalent
  circuit models for Li-ion batteries." *Journal of Power Sources*, 198.
- Hu, X., Feng, F., et al. (2024). "Fault-tolerant state estimation for
  lithium-ion batteries: a review." *IEEE Transactions on Transportation
  Electrification*.
- Lipu, M. S. H., et al. (2024). "Recent advances in transformer-based
  state of charge estimation for lithium-ion batteries." *Journal of Energy
  Storage*.
- Liu, K., Ashwin, T. R., Hu, X., Lucu, M., Widanage, W. D. (2019). "An
  evaluation study of different modelling techniques for calendar ageing
  prediction of lithium-ion batteries." *Renewable and Sustainable Energy
  Reviews*.
- Liu, Y., et al. (2022). "Hybrid physics-data driven approach for state
  of charge estimation of lithium-ion batteries." *Applied Energy*.
- Ma, Y., Zhou, X., Li, B., Chen, H. (2018). "Comparative study of
  non-electrochemical hysteresis models for LiFePO4/graphite batteries."
  *Journal of Power Electronics*, 18(5).
- Marongiu, A., Roscher, M. A. (2015). "Influence of the hysteresis effect
  in LiFePO4 batteries on usable capacity and accuracy of state-of-charge
  estimation." *Energy Procedia*.
- Moura, S. J., Argomedo, F. B., Klein, R., Mirtabatabaei, A., Krstic, M.
  (2017). "Battery state estimation for a single particle model with
  electrolyte dynamics." *IEEE Transactions on Control Systems Technology*.
- Peng, J., Luo, J., He, H., Lu, B. (2019). "An improved state of charge
  estimation method based on cubature Kalman filter for lithium-ion battery."
  *Applied Energy*.
- Plett, G. L. (2004). "Extended Kalman filtering for battery management
  systems of LiPB-based HEV battery packs — Parts 1–3." *Journal of Power
  Sources*, 134.
- Plett, G. L. (2015). *Battery Management Systems, Volume II:
  Equivalent-Circuit Methods*. Artech House.
- Roscher, M. A., Sauer, D. U. (2011). "Dynamic electric behavior and
  open-circuit-voltage modeling of LiFePO4-based lithium-ion secondary
  batteries." *Journal of Power Sources*.
- Sun, F., Hu, X., Zou, Y., Li, S. (2014). "Adaptive unscented Kalman
  filtering for state of charge estimation of a lithium-ion battery for
  electric vehicles." *Energy*.
- Tian, J., Xiong, R., Shen, W. (2020). "A review on state of health
  estimation for lithium ion batteries in photovoltaic systems."
  *eTransportation*.
- Tian, Y., et al. (2022). "Fault-tolerant state-of-charge estimation for
  lithium-ion battery packs under sensor faults." *IEEE Transactions on
  Power Electronics*.
- Vidal, C., Malysz, P., Kollmeyer, P., Emadi, A. (2020). "Machine
  learning applied to electrified vehicle battery state of charge and
  state of health estimation: State-of-the-art." *IEEE Access*, 8.
- Wang, Y., Tian, J., Sun, Z., Wang, L., Xu, R., Li, M., Chen, Z. (2020).
  "A comprehensive review of battery modeling and state estimation
  approaches for advanced battery management systems." *Renewable and
  Sustainable Energy Reviews*.
- Wang, Y., et al. (2021). "Saturation-based state of charge estimation
  for LiFePO4 batteries with flat OCV characteristics." *Journal of Energy
  Storage*.
- Xiong, R., Cao, J., Yu, Q., He, H., Sun, F. (2017). "Critical review on
  the battery state of charge estimation methods for electric vehicles."
  *IEEE Access*, 6.
- Xiong, R., Yu, Q., Wang, L. Y., Lin, C. (2018). "A novel method to
  obtain the open circuit voltage for the state of charge of lithium-ion
  batteries in electric vehicles by using H infinity filter." *Applied
  Energy*.
