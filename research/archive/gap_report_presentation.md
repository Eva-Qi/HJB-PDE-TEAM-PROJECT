# MF796 Project — Report & Presentation Gap Analysis
**Date:** 2026-03-27
**Covers:** Report structure, key visualizations, 10-minute presentation strategy

---

## PART 1: How to Structure a 10-Page Computational Finance Report

### Standard Section Layout (recommended page allocation)

| Section | Pages | Content |
|---|---|---|
| Abstract | 0.25 | 4–5 sentences: problem, method, key results, contribution |
| 1. Introduction | 0.75 | Why liquidation matters, BTC context, gap in literature |
| 2. Model & Theory | 2.0 | AC framework equations, HJB derivation sketch, Heston extension |
| 3. Data & Calibration | 1.0 | Binance BTCUSDT, GK estimator, Kyle's lambda methodology |
| 4. Numerical Methods | 2.0 | PDE solver (Riccati ODE + policy iteration), MC engine design |
| 5. Results & Analysis | 3.0 | Trajectory plots, cost distributions, convergence, regime |
| 6. Conclusion | 0.5 | What was achieved, limitations, future work (Heston/regime) |
| References | 0.5 | ~10–15 citations |

**Total: 10 pages**

### Math Derivation vs. Results Balance

The MF796 course (QST MF 796, BU Questrom) emphasizes "computational requirements and trade-off between computational effort and accuracy" — this signals the professor wants to see:
- **Key equations stated but NOT fully re-derived**: Show the HJB and the Riccati reduction, state the closed-form trajectory. Do not reproduce the full Almgren-Chriss (2000) paper.
- **Proof-of-concept derivation**: One page showing how the HJB reduces to the Riccati ODE is enough. Skip full stochastic control proofs.
- **Numerical results dominate**: Sections 4 and 5 together should be the longest (5 pages combined). Figures carry the argument.
- **Rule of thumb**: ~40% math/model, ~60% results/discussion for a 10-page project report (vs. a pure theory paper which flips this).

### Convergence Study — How to Present It

Include a convergence table and/or plot in Section 4:
- **Table**: Grid size N × M vs. L2/L∞ error (compare to analytical solution if available, or Richardson extrapolation)
- **Plot**: Log-log plot of error vs. grid spacing — slope of ~2 confirms second-order scheme (Crank-Nicolson) or ~1 for first-order
- **Key metric to report**: CPU time vs. accuracy tradeoff — this is exactly what MF796 course rubric emphasizes
- Label the "good enough" operating point used for production runs

### Strategy Comparison — How to Present It

Present in Section 5 using a dedicated subsection "Execution Strategy Comparison":
- Show TWAP (constant rate), VWAP (volume-weighted), and Optimal AC trajectory on the same plot
- Report a comparison table: Expected Cost (E[IS]), Variance Var[IS], VaR 95%, CVaR 95% for each strategy
- One or two sentences per row explaining the economic intuition of the difference

### Code Detail Level

- Do NOT include code blocks in the report body
- In an Appendix (if page limit allows), list: language, key libraries, GitHub repo link or attach as supplement
- Reference your numerical scheme choices inline (e.g., "we discretize using Crank-Nicolson with N=100 time steps")
- Mention test coverage (53 tests passing) as a sentence in Section 4 to demonstrate rigor

---

## PART 2: Visualizations That Make the Strongest Impression

Priority order: highest impact first. Target 6–8 figures total for a 10-page report.

### Figure 1 (REQUIRED): Trading Trajectory Comparison
**What it shows:** Inventory x(t) vs. time for three strategies on the same axes
- Curve A: Risk-averse optimal (lambda > 0) — front-loaded, concave decay
- Curve B: TWAP — straight diagonal line (baseline)
- Curve C: Risk-neutral (lambda = 0) — uniform speed
- Optionally: one VWAP curve

**Why it works:** Instantly communicates the core result. The hyperbolic sine shape of the AC optimal is visually distinctive versus the TWAP line. Audience grasps intuitively that urgency increases near T=0.

**Styling notes:** Use thick lines with distinct colors. Add vertical dashed line at T. Label lambda values on each curve. X-axis: time (hours or normalized), Y-axis: shares remaining.

### Figure 2 (REQUIRED): Efficient Frontier
**What it shows:** Expected cost E[IS] vs. Variance Var[IS] as a parametric curve in lambda
- Mark specific strategies A (risk-averse), B (risk-neutral), C (risk-seeking) as labeled dots
- Show TWAP and VWAP as off-frontier reference points (dashed markers)

**Why it works:** The AC efficient frontier is the model's single most iconic result. Showing TWAP sits off the frontier immediately communicates the optimality claim. Calibrated to your BTCUSDT parameters (sigma=42.1%, gamma=0.0133) makes it original.

**Styling notes:** Annotate "minimum variance" and "minimum cost" endpoints. Use gray shading above the frontier to indicate dominated strategies.

### Figure 3 (REQUIRED): Execution Cost Distribution with VaR/CVaR
**What it shows:** Histogram of MC-simulated implementation shortfall for the optimal strategy
- Vertical dashed lines marking VaR 95% and CVaR 95%
- Optionally overlay TWAP cost distribution in a lighter color

**Why it works:** Translates abstract risk measures into a concrete picture. The fat right tail and the CVaR exceeding VaR visually explains why CVaR is a superior risk measure. Shows the MC engine output directly.

**Styling notes:** 1000+ paths minimum for smooth histogram. Use log scale on y-axis if tail is thin. Annotate E[cost], VaR, CVaR with arrows. Use seaborn `histplot` or matplotlib with KDE overlay.

### Figure 4 (REQUIRED): Sensitivity Heatmap — Cost vs. (lambda, sigma)
**What it shows:** 2D heatmap where x-axis = risk aversion lambda, y-axis = volatility sigma, color = expected execution cost
- Add a contour line showing "equal cost" curves

**Why it works:** Single figure answers "how sensitive is the strategy to our calibration uncertainty?" This is the key robustness check. Interviewers and professors love parameter sensitivity — it shows you understand the model is as only good as its inputs.

**Styling notes:** Use `matplotlib imshow` or `seaborn heatmap`. Mark the calibrated operating point (lambda=0.0133, sigma=42.1%) with a star. Add colorbar with cost units.

### Figure 5 (RECOMMENDED): PDE Convergence Plot
**What it shows:** Log-log plot of numerical error vs. grid spacing h (or number of grid points N)
- Two curves: time refinement and space refinement
- Reference slope lines: dashed lines with slope 1 and slope 2 for visual comparison

**Why it works:** Directly demonstrates the solver works correctly. A clean order-2 convergence line is the most convincing validation of a PDE solver. Professors from a numerical methods course will look for this.

**Styling notes:** Use Richardson extrapolation estimate as "true" solution if no analytical solution available. Mark the "production" grid size with a vertical dashed line.

### Figure 6 (RECOMMENDED): Regime Detection Overlay
**What it shows:** BTC price time series with regime labels shaded in background
- Background: green shading for low-vol regime, red/orange for high-vol regime
- Overlay: two optimal trajectory curves (one per regime) shown in inset or as a second panel

**Why it works:** The most visually compelling figure in the set. Ties the abstract regime model to real price data. Immediately intuitive that execution strategy should differ between regimes.

**Styling notes:** Use matplotlib `axvspan` for regime shading. If regime detection is not fully implemented, use a hand-labeled subset of BTCUSDT data as a placeholder. Bottom panel: show volatility estimate (rolling std) to justify regime boundaries.

### Figure 7 (OPTIONAL): MC Convergence — Standard Error vs. Path Count
**What it shows:** Standard error of E[IS] estimate vs. number of MC paths N
- Curves for: naive MC, antithetic variates, control variate
- Expected slope: -0.5 on log-log plot (CLT convergence rate)
- Antithetic and CV curves should be shifted down (lower SE for same N)

**Why it works:** Directly demonstrates value of variance reduction methods (antithetic + CV). Shows your MC engine is correctly implemented. The three curves separating on the plot is a clean quantitative result.

### Figure 8 (OPTIONAL): Heston vs. GBM Trajectory Comparison
**What it shows:** Under Heston dynamics, plot expected cost and variance for a range of lambda values, overlaid against GBM results
- Shows how stochastic vol changes the efficient frontier shape
- Highlight the regime where Heston materially differs from GBM

**Why it works:** Directly motivates the Heston extension. If the frontiers overlap completely, the extension adds no value — if they diverge, it does. Either result is scientifically interesting.

---

## PART 3: 10-Minute Presentation Strategy

### Slide Count and Time Budget

**Target: 10–12 slides total** (1 slide per minute; a few slides go faster)

| Slide | Time | Content |
|---|---|---|
| 1. Title | 0:00–0:30 | Title, names, one-line hook ("How do you sell $10M of BTC without moving the market?") |
| 2. Motivation | 0:30–1:30 | Market impact problem, real-world cost of naive execution, 2 bullet points |
| 3. The AC Model (math) | 1:30–3:00 | 3 key equations: dynamics, cost functional, optimal trajectory. No more. |
| 4. Calibration from BTC data | 3:00–4:00 | σ=42.1%, γ=0.0133, one data plot or table. State the method, not the derivation. |
| 5. Numerical Methods | 4:00–5:00 | HJB → Riccati ODE, MC engine diagram. 2 bullets each. |
| 6. Result: Trajectories | 5:00–6:00 | Figure 1 (trajectory comparison). Spend full minute explaining what the curves mean. |
| 7. Result: Efficient Frontier | 6:00–6:45 | Figure 2. "TWAP is off the frontier — optimal strategy strictly dominates." |
| 8. Result: Cost Distribution | 6:45–7:30 | Figure 3 (histogram + VaR/CVaR). Explain CVaR > VaR intuition in one sentence. |
| 9. Sensitivity & Robustness | 7:30–8:15 | Figure 4 (heatmap). "Our calibrated point lies in a stable region." |
| 10. Extensions (Heston + Regime) | 8:15–9:00 | One slide showing the extension framework. State research questions, not full results. |
| 11. Conclusion | 9:00–9:30 | 3 bullets: what was done, key finding, one open question |
| 12. Q&A / backup | 9:30–10:00 | "Thank you" + one backup slide with convergence table ready |

### Explaining Almgren-Chriss to a Mixed Audience (the first 2 minutes)

Use this three-sentence structure on slide 2 or 3:

> "When you need to liquidate a large position quickly, buying and selling your own shares moves the price against you — this is market impact. Almgren and Chriss (2000) showed that there is an optimal schedule: sell faster early to reduce risk, but not so fast that impact costs dominate. We implement and calibrate this model on live Bitcoin data."

This requires zero prior knowledge and frames the tradeoff immediately. Then show the closed-form trajectory formula on slide 3 — it is short enough to fit in one line.

### Slides vs. Verbal Content — What Goes Where

**Put on slides:**
- Key equations (no more than 3 per slide, font size >= 20pt)
- All figures (one figure per slide, full-size)
- Section headers as complete declarative sentences (e.g., "Optimal trajectories front-load execution" not "Results")
- A 2-column comparison table for strategy results

**Say verbally (NOT on slides):**
- Intuition and economic interpretation of each figure
- Why you made specific modeling choices (why GK estimator, why antithetic variates)
- Limitations of the model
- Connection to real trading desk practice

**Rule:** If a sentence is on the slide and you are also saying it, one of those is redundant. Reserve slide text for definitions, formulas, and quantitative results only.

### Live Code Demo: Do Not Do It

For a 10-minute academic presentation:
- **Do not do a live demo.** 10 minutes is not enough time to context-switch between slides and a terminal.
- If you want to reference implementation, include a 2-line code snippet on a slide (e.g., the Riccati ODE solver call signature) — this is more controlled and reproducible.
- Pre-recorded short screencasts (15–30 seconds) embedded in slides are acceptable if the visual is genuinely informative, but for this topic the figures carry more weight than watching code run.
- Exception: if the professor explicitly asks for a demo, prepare a single Jupyter notebook cell that runs in < 5 seconds and produces one clean figure.

### Practice Protocol

- Do at least 2 full timed run-throughs aloud (not in your head — speaking reveals problems mental rehearsal does not)
- Record yourself once — playback will identify slides you spend too long on
- The most common failure mode: spending 3 minutes on model setup and having 2 minutes for results. Invert this — results are the deliverable.
- Prepare 2–3 backup slides: full convergence table, parameter table (all calibrated values), extended sensitivity analysis. Do not show them unless asked.

### How to Handle "We ran out of time for Part D/E" Honestly

On slide 10 (Extensions), frame it as:
- "Parts D (Heston) and E (Regime Detection) represent the research frontier of this project. We completed the theoretical framework and model design; full numerical results are in progress."
- Do NOT apologize. Incomplete extensions are normal in term projects. State what you designed and what the expected result would be.

---

## Summary: Priority Checklist Before April 28

### Report
- [ ] Write Sections 1–3 (model + calibration) with equations from existing code
- [ ] Write Section 4 using the 53 passing tests as evidence of correctness
- [ ] Produce Figures 1–5 (the required ones above) — at minimum these 5
- [ ] Add convergence table to Section 4
- [ ] Write Section 6 framing Heston/Regime as future work

### Presentation
- [ ] Build 11-slide deck using figures from the report
- [ ] Turn every slide title into a declarative claim
- [ ] Time two full run-throughs; ensure results get >= 4 minutes
- [ ] Prepare 2 backup slides (convergence table, parameter summary)

---

## Sources Consulted

- [Optimal Execution of Portfolio Transactions — Almgren & Chriss (2000)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Deep Dive into IS: The Almgren-Chriss Framework — Anboto Labs (Medium)](https://medium.com/@anboto_labs/deep-dive-into-is-the-almgren-chriss-framework-be45a1bde831)
- [Solving the Almgren-Chriss Model — Dean Markwick](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html)
- [Trading Execution Algorithms: The AC Framework — Ibrahim Adedimeji (Medium, Feb 2026)](https://medium.com/@ibrahimlanre1890/trading-execution-algorithms-the-almgren-chriss-framework-56717dd650ce)
- [A Tale of Two Models: Implementing AC via Nonlinear and Dynamic Programming — Bagourd (2022)](https://www.arthur.bagourd.com/wp-content/uploads/2022/08/A_Tale_of_Two_Models__Implementing_the_Almgren_Chriss_framework_through_nonlinear_and_dynamic_programming.pdf)
- [Numerical Methods for Controlled HJB PDEs in Finance — Forsyth (Waterloo)](https://cs.uwaterloo.ca/~paforsyt/hjb.pdf)
- [Ten Simple Rules for Effective Presentation Slides — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8638955/)
- [10 Tips for the 10-Minute Conference Presentation — UBC Science Writing](https://scwrl.ubc.ca/2016/07/21/10-tips-for-the-10-minute-conference-presentation/)
- [How to Give a Technical Presentation — Michael Ernst, UW](https://homes.cs.washington.edu/~mernst/advice/giving-talk.html)
- [QST MF 796 Course Description — Boston University](https://www.bu.edu/academics/questrom/courses/qst-mf-796/)
- [Presentation Patterns: Demonstrations vs. Presentations — InformIT](https://www.informit.com/articles/article.aspx?p=1930512)
- [Writing Quantitative Research Reports — UF Writing in the Disciplines](https://portal.clas.ufl.edu/writing--wdkb-v1/assignment/quantitative-research-reports/)
- [IMRaD Report Writing — GMU Writing Center](https://writingcenter.gmu.edu/writing-resources/imrad/writing-an-imrad-report)
- [Monte Carlo Methods for VaR and CVaR — ACM TOMS (2014)](https://dl.acm.org/doi/10.1145/2661631)
