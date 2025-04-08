
# Voluntary Carbon Market in Agriculture: A Game-Theoretic and Mechanism Design Analysis

`Game Theory 2025 - Mini Project`
---

**Author:** Yalla Mahanth (mahanthyalla[at]iisc[dot]ac[dot]in)

**Course:** E1 254 Game Theory and Mechanism Design, IISc Bangalore

**Report:** [Game Theory and Mechanism Design for Voluntary Carbon Market in Agriculture](will_host_it_soon)

## Project Overview

This project invouled in design and implementation of a Voluntary Carbon Market (VCM) tailored for India's agricultural sector, focusing on overcoming participation barriers for small and marginal farmers. Agriculture is vital to India's economy and food security but faces challenges from climate change and contributes to greenhouse gas emissions. The Government of India's proposed VCM framework \[[MoAFW2024VCM](#references)] aims to incentivize sustainable practices (like agroforestry, soil carbon enhancement) by allowing farmers to generate and sell carbon credits.

However, smallholder farmers (over 86% of Indian farmers) face significant hurdles like high transaction costs for verification, limited resources, and low bargaining power. The government framework suggests using Farmer Producer Organizations (FPOs) for aggregation. This project uses analytical tools to address critical questions arising from this framework:

1.  **How can FPOs operate effectively and stably?**
2.  **How should the collective benefits (carbon revenue) be fairly distributed among participating farmers?**
3.  **What market mechanisms are suitable for trading these credits, ensuring fairness and encouraging farmer participation?**

We employ a dual approach:
*   **Cooperative Game Theory:** To model FPOs as coalitions, analyze stability (Core concept), and evaluate fair allocation rules (Shapley Value).
*   **Mechanism Design:** To analyze and compare credit trading mechanisms (VCG Auction, Uniform Price Auction) against cooperative solutions based on efficiency, fairness, and crucially, participation incentives (Individual Rationality).

The analysis uses computational simulations based on synthetic farmer data reflecting realistic heterogeneity.

## Key Concepts

*   **Voluntary Carbon Market (VCM):** A market where entities voluntarily buy carbon credits to offset their emissions. Credits are generated from projects that verifiably reduce or remove greenhouse gases (e.g., sustainable agriculture).
*   **Carbon Credits:** A tradable certificate representing the reduction or removal of one metric tonne of CO2 equivalent (tCO2e).
*   **Farmer Producer Organization (FPO):** A legal entity formed by primary producers (farmers) to undertake collective business activities, including input procurement, production, and marketing. In the VCM context, they act as aggregators.
*   **Cooperative Game Theory:** Analyzes situations where players (farmers) can form binding agreements (coalitions/FPOs) to achieve joint benefits.
*   **Characteristic Function ($v(S)$):** A function defining the total value a coalition $S$ can achieve by cooperating. In our model: 
    $$ \displaystyle v(S) = \alpha  \sum(r_i) + \beta (\sum(r_i))^2 $$
    *   $r_i$: Farmer $i$'s baseline standalone payoff.
    *   $\alpha$: Baseline scaling factor.
    *   $\beta$: Synergy factor representing non-linear benefits of scale.
*   **The Core:** A solution concept in cooperative games. An allocation is in the Core if no subgroup of players can achieve a better outcome by splitting off from the grand coalition. Ensures stability.
*   **Shapley Value ($\phi_i(v)$):** A unique, axiomatically fair method to distribute the total value of a coalition among its members based on their average marginal contribution.
*   **Mechanism Design:** The art of designing the "rules of the game" (e.g., auction rules) to achieve desired outcomes (efficiency, fairness, truthfulness) when participants act strategically.
*   **VCG Auction (Vickrey-Clarke-Groves):** An auction mechanism known for efficiency (maximizing total surplus) and incentive compatibility (truthful bidding is optimal) under certain assumptions. Winners pay based on the externality they impose.
*   **Individual Rationality (IR):** A participation constraint. In this context, specifically refers to whether a farmer's payoff from participating in the VCM (via FPO/auction) $x_i$ is at least as good as their standalone farming payoff $r_i$ (i.e., $x_i \ge r_i$). This is critical for *voluntary* participation.
*   **Gini Coefficient:** A statistical measure of distribution inequality (0 = perfect equality, 1 = maximum inequality). Used here to measure the fairness of payoff distributions among farmers.

## Methodology

### 1. Synthetic Data Generation
*   A dataset simulating 250 heterogeneous Indian farmers was generated.
*   Key attributes per farmer ($i$):
    *   Baseline standalone payoff ($r_i$): Sampled from Normal(20000, 5000).
    *   Potential carbon credits ($q_i$): Sampled from Gamma(2.5, 1.8).
    *   True cost per credit ($c_i$): Sampled from Gamma(3, 800) + 500.
    *   Other attributes (farm size, demographics, risk aversion) based on plausible distributions (see `generate_data.py` for details).
*   This dataset provides the input parameters ($r_i, q_i, c_i$) for the game-theoretic and mechanism design models.

### 2. Cooperative Game Model (FPOs)
*   FPOs are modeled as coalitions $S$ within the set of farmers $N$.
*   The value generated $v(S)$ is calculated using the characteristic function 
$$
    v(S) = \alpha \sum(r_i) + \beta (\sum(r_i))^2
$$ 
  
Parameters $\alpha (\texttt{alpha})$ and $\beta (\texttt{beta})$ were varied across experiments.
*   **Shapley Value Calculation:** Computed using either the exact permutation method (for $N <= 10$) or Monte Carlo sampling (for $N > 10$, typically 10,000 samples) to determine fair payoffs $\phi_i(v)$. See `utils.py`.
*   **Core Stability Check:** Implemented by checking the Core conditions for all non-trivial subsets (feasible only for $N <= 15$). An allocation $x$ is stable if 
  $$ \displaystyle \sum(x_i) = v(N) $$ 
  and 
  $$ \displaystyle \sum_{i \in S} x_i >= v(S) $$ 
  for all $S$. See `utils.py`.

### 3. Mechanism Design Models (Trading)
*   **VCG Auction:** Implemented as described in the [Report](#references) (Section 3.3, Algorithm 3.3). Sellers (farmers) are assumed to bid their true cost $c_i$. Winners are those with $c_i <= p_{max}$ (market price). Payment $P_i$ is based on the critical cost (lowest losing bid or $p_{max}$). See `mechanism.py`.
*   **Uniform Price Auction:** Implemented based on sorting sellers by cost $c_i$, fulfilling a fixed buyer demand $Q_{demand}$, and setting a single clearing price based on the first excluded seller. See `mechanism.py`.

### 4. Evaluation Metrics
*   Key metrics calculated include: Average Farmer Payoff, Absolute/Percentage Gain ($x_i - r_i$), IR Met Percentage ($x_i >= r_i$), Gini Coefficient, Core Stability Status, VCG Surplus, VCG Budget Balance, Number of Winners, Total Credits Supplied. See `metrics.py`.

## Experimental Scenarios and Setup

Various simulation scenarios were conducted to analyze different facets:

*   **Coalition Analysis (N=3 to 12):** Examined the effect of FPO size on average value and Shapley payoff per farmer. Assessed Core stability of Shapley, Equal Split, and Proportional allocations for N=12.
*   **Pricing Analysis (N=250):** Simulated VCG auctions across a wide range of market prices ($p_{max}$ from 500 to 4000 INR) to generate supply curves and analyze VCG performance metrics (surplus, payments, fairness, etc.).
*   **Mechanism Comparison (N=12 to 250):** Directly compared Shapley Allocation (with baseline $alpha=1.0, beta=0.0$), VCG Auction, and Uniform Price Auction across varying market prices. Focused on Average Farmer Profit, Gini, and crucially, IR Met % (vs. $r_i$).
*   **Parameter Sensitivity (N=100):** Systematically varied $\alpha$ (0.75-1.50, with $beta=0$) and $\beta$ (0.0-0.2, with fixed $\alpha$) to understand their impact on Shapley payoffs and IR satisfaction.
*   **Heterogeneity Analysis (N=15):** Simulated a mixed coalition of `Small` and `Large` farmers to analyze if Shapley allocation provides equitable relative gains.
*   **Aggregator Model Analysis (N=15, N=250):** Explored the novel extension where an aggregator incurs costs ($C_{base}$, $C_{var}$) and takes a commission ($delta$), analyzing the impact on net farmer payoffs ($V_F(S)$), IR, aggregator profit, and stability.

## Results and Discussion

1.  **Value of Aggregation:** Simulations confirm that FPOs significantly increase the average value and potential payoff per farmer compared to standalone operation, especially when synergistic benefits ($beta > 0$) are present. Larger coalitions yield higher average value (See [Report](#references) Fig 5.1).

2.  **Shapley Value Performance:**
    *   **Fairness:** Consistently provides equitable distributions with low Gini coefficients (~0.12-0.14) across various scenarios (See [Report](#references) Fig 5.4, 5.8).
    *   **Individual Rationality (vs. $r_i$):** Crucially, Shapley meets the participation IR constraint (100% farmers have $phi_i >= r_i$) provided the net benefit from cooperation is non-negative (i.e., $alpha >= 1$ if $beta=0$, or $beta > 0$). This is vital for voluntary adoption (See [Report](#references) Fig 5.3).
    *   **Stability:** For small N (N=12, N=15), Shapley allocations were found to be in the Core under tested parameters, suggesting stable cooperation (See [Report](#references) Table 5.1).

3.  **Auction Mechanism Performance:**
    *   **VCG Efficiency:** Maximizes social surplus, and participation increases with market price (See [Report](#references) Fig 5.2, 5.6). Guarantees winners cover their carbon generation cost ($c_i$).
    *   **VCG Participation Incentive Failure:** VCG (and Uniform Price) perform poorly on the critical IR metric ($x_i >= r_i$). Payoffs often don't compensate for the baseline farming opportunity cost, potentially discouraging voluntary participation (See [Report](#references) Fig 5.3).
    *   **VCG Budget:** Generates a significant surplus for the auctioneer (See [Report](#references) Fig 5.6).

4.  **Heterogeneity Impact:** Shapley value allocates gains proportionally to baseline contribution ($r_i$). In simulations with mixed small/large farmers, percentage gains were similar or identical, suggesting both groups have incentives to join (See [Report](#references) Fig 5.5).

5.  **Parameter Sensitivity:** Outcomes are highly sensitive to $\alpha$ and $\beta$. $\alpha < 1$ (with $\beta=0$) fails IR. Small positive $\beta$ creates large gains due to the quadratic term, highlighting the need for realistic estimation.

6.  **Aggregator Model Insights:** Introducing aggregator costs ($C_A(S)$) and commission ($delta$) reduces net farmer payoffs. There exists a critical $delta$ threshold (~15-20% in simulations) above which farmer participation (IR vs. $r_i$) collapses, even if the remaining allocation is Core-stable. This demonstrates the trade-off between aggregator viability and farmer incentives (See [Report](#references) Fig A.1).

## Novel Extension: Aggregator as a Strategic Player (Appendix A)

*   **Model:** Explicitly includes an aggregator $A$ with costs 
$$C_A(S) = C_{base} + C_{var} * |S|$$ 
and a commission rate $delta$. The net value available to farmers is $V_F(S) = (1 - delta) * max(0, V(S) - C_A(S))$.
*   **Methodology:** Simulated this model for N=15 and N=250, varying $delta$ from 0% to 50%. Calculated Shapley payoffs $\phi_i(V_F)$, aggregator profit, IR%, Gini, and Core stability (N=15).
*   **Key Finding:** Identified a critical commission threshold (~15-20% in these runs) beyond which farmer participation collapses ($IR\% \to 0$), demonstrating the crucial need to balance aggregator revenue with farmer incentives. Shapley maintained fairness (low Gini) and Core stability (for N=15) across commission rates, but participation failure renders stability moot.

## Conclusions

1.  Aggregation via FPOs is crucial for smallholder participation in agricultural VCMs.
2.  The Shapley value provides a demonstrably fair and stable (for small N) allocation mechanism that, crucially, satisfies the individual rationality constraint necessary for voluntary participation, provided the cooperative effort yields a net positive return over baseline farming.
3.  Standard auction mechanisms like VCG, while efficient, may fail to incentivize broad voluntary farmer participation if payoffs don't sufficiently exceed baseline farming income ($r_i$).
4.  The benefits of cooperation are highly sensitive to modeling parameters ($\alpha$, $\beta$); realistic estimation is key.
5.  Explicitly modeling aggregator costs and commissions reveals a critical trade-off: excessive commission rates destroy farmer participation incentives, regardless of the fairness of internal allocation.
6.  Game theory and mechanism design provide essential tools for analyzing and designing effective, equitable VCMs tailored to the complexities of Indian agriculture.

## Repository Structure

*(See detailed structure listing in the initial sections of this README)*

## Setup and Installation

1.  **Clone:** 
    ```bash 
    git clone https://github.com/Mahanth-Maha/GameTheory2025MiniProject.git
    ```
2.  **Navigate:** 
    ```bash 
    cd GameTheory2025MiniProject
    ```
3.  **Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
4.  **Install Packages:**
    ```bash
    pip install pandas numpy matplotlib seaborn tqdm
    ```

## Running the Experiments

Execute scripts from the main project directory. Key scripts and examples:

*   **Generate Data:** 
    ```python
    python3 generate_data.py
    ```
*   **Coalition Analysis (N=12):** 
    ```python
    python3 01_Analysis_Coalition.py -n 12
    ```
*   **Pricing Analysis (N=250):** 
    ```python
    python3 02_Analysis_Pricing.py -n 250
    ```
*   **Mechanism Comparison (N=250):** 
    ```python
    python3 mechanism_search.py -n 250 --output_csv ./data/synthetic/comparison_results.csv
    ```
*   **Parameter Sensitivity (Alpha=1.1, Beta=0.0, N=100):** 
    ```python
    python3 03_best_mechanism.py -n 100 --alpha 1.1 --beta 0.0
    ```
*   **Heterogeneity Analysis:** 
    ```python
    python3 04_small_vs_large_farmers.py
    ```
*   **Aggregator Model:** 
    ```python
    python3 05_Aggregator_as_a_player.py
    ```

*(See individual scripts or use `-h` for more command-line options like input file path, plot directories, specific parameters, etc.)*


# references 

*   [MoAFW2024VCM](https://www.mofw.gov.in/sites/default/files/Voluntary%20Carbon%20Market%20Framework%20for%20India.pdf) - Government of India, Ministry of Agriculture and Farmers Welfare, 2024. Voluntary Carbon Market Framework for India.*
*   [Report](will_host_it_soon)   
