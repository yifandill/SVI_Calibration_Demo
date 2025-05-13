# SVI Calibration

In financial modelling, calibration refers to the process of adjusting the parameters of a model so that its outputs match observed market prices as closely as possible. In case of the SVI model, its parameterization is as follows:

$$\omega(k) = a + b\left\{\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right\}$$

where $\omega$ is the implied total variance and $k$ is the log forward moneyness, $a$, $b$, $\rho$, $m$, $\sigma$ are the five parameters used to fit $\omega$ on a given expiry for any $k$. For more details, refer to the seminal paper on SVI model, *Gatheral, Jim, and Antoine Jacquier. "Arbitrage-free SVI volatility surfaces." Quantitative Finance 14.1 (2014): 59-71*.

Remark: Let $S_0$ be the spot price today. $r$ is the continuously‑compounded risk‑free rate, and $q$ is the continuous dividend rate. The forward price for maturity $t$ is $F:=F_{0,t} = S_0 e^{(r-q)t}$. Let the strike of the option be $K$. The log forward moneyness is given by $k=\log \frac{K}{F}$.

## Requirements

1. Write an SVI calibrator
Develop an SVI calibrator to solve for the SVI parameters against the points on an implied volatility smile. To help you get started, we have provided sample code in demo_svi_calibrator.txt (some modifications are needed to get it running). Alternatively, you may find sample code on github [SVI-Volatility-Surface-Calibration](https://github.com/wangys96/SVI-Volatility-Surface-Calibration).

2. Propose quantitative measures to evaluate the quality of the calibrator
The calibrator in 1 may be able to find the SVI parameters such that the “calibration” error is minimized. However, this does not guarantee a satisfactory calibration result. For example, matching an outlier might compromise the fit of the entire curve, and matching the wings might trade off against accurately fitting the near-the-money points

3. Stress data test, 2025-04-08, AAPL.US
On a typical market turmoil selloff day, where the market trades towards the put wings (downward skewing), verify if the calibrator in 1 can handle this scenario correctly. For example, ensure it does not result in negative volatilities in the right asymptote.

4. Day to day parameter stability
Utilize real data in data set folder to test the day-to-day stability of the calibrator in 1. Quantify the stability measure and propose improvement methods if it does not perform well. For example, refer to the paper by *Ferhati, Tahar, titled 'Robust Calibration for SVI Model Arbitrage Free,' available at SSRN 3543766 (2020)*, for guidance on setting appropriate parameter boundaries for the SVI-Quasi-explicit calibrator.

## Analysis

Quoted from *Gatheral, Jim, and Antoine Jacquier. "Arbitrage-free SVI volatility surfaces." Quantitative Finance 14.1 (2014): 59-71*.

### Setting

Consider a stock price process $\left(S_t\right)_{t \geq 0}$ with natural filtration $\left(\mathcal{F}_t\right)_{t \geq 0}$, and we define the forward price process $\left(F_t\right)_{t \geq 0}$ by $F_t:=\mathbb{E}\left(S_t \mid \mathcal{F}_0\right)$. For any $k \in \mathbb{R}$ and $t>0, C_{\mathrm{BS}}\left(k, \sigma^2 t\right)$ denotes the Black-Scholes price of a European Call option on $S$ with strike $F_t \mathrm{e}^k$, maturity $t$ and volatility $\sigma>0$. We shall denote the Black-Scholes implied volatility by $\sigma_{\mathrm{BS}}(k, t)$, and define the total implied variance by

$$
w(k, t)=\sigma_{\mathrm{BS}}^2(k, t) t .
$$

We shall refer to the two-dimensional map $(k, t) \mapsto \omega(k, t)$ as the volatility surface, and for any fixed maturity $t > 0$, the function $k \mapsto \omega(k, t)$ will represent a slice. For a given maturity slice, we shall use the notation $\omega(k):=\omega(k; \chi)$ where $\chi$ represents a set of parameters, and drop the $t$-dependence. Unless otherwise stated, we shall always assume that the map $k \mapsto w(k, t)$ is at least of class $\mathcal{C}^2(\mathbb{R})$ for all $t \geq 0$.

### No Arbitrage Condition

**Definition 2.1.** A volatility surface is free of static arbitrage if and only if the following conditions are satisfied:

- it is free of calendar spread arbitrage;
- each time slice is free of butterfly arbitrage.

**Definition 2.2.** A volatility surface $\omega$ is free of calendar spread arbitrage if

$$
\partial_t \omega(k,t)\geq 0,\, \forall k\in\mathbb{R}\, ,t>0.
$$

**Lemma 2.2.** A slice is free of butterfly arbitrage if and only if $g(k)\geq 0$ for all $k\in\mathbb{R}$ and $\lim_{k\to+\infty}d_{+}(k) = -\infty$, where $d_{\pm}(k) := -k/\sqrt{\omega(k)}\pm \sqrt{\omega(k)}/2$, and $g:\mathbb{R}\to\mathbb{R}$ is defined by

$$
g(k):=\left(1-\frac{k \omega^{\prime}(k)}{2 \omega(k)}\right)^2-\frac{\omega^{\prime}(k)^2}{4}\left(\frac{1}{\omega(k)}+\frac{1}{4}\right)+\frac{\omega^{\prime \prime}(k)}{2}.
$$

### SVI Formulation

For a given parameter set $\chi_R=\{a, b, \rho, m, \sigma\}$, the *raw SVI parameterization* of total implied variance reads:

$$
\omega\left(k ; \chi_R\right)=a+b\left\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\right\}
$$

where $a \in \mathbb{R}, b \geq 0,|\rho|<1, m \in \mathbb{R}, \sigma>0$, and the obvious condition $a+b \sigma \sqrt{1-\rho^2} \geq 0$, which ensures that $\omega\left(k ; \chi_R\right) \geq 0$ for all $k \in \mathbb{R}$. This condition indeed ensures that the minimum of the function $\omega\left(\cdot ; \chi_R\right)$ is non-negative. Note further that the function $k \mapsto \omega\left(k ; \chi_R\right)$ is (strictly) convex on the whole real line.

We exclude the trivial cases $\rho = 1$ and $\rho = −1$, where the volatility smile is respectively strictly increasing and decreasing. We also exclude the case $\sigma = 0$ which corresponds to a linear smile.

## Robust Calibration For SVI Arbitrage Free

Quoted from *Ferhati, Tahar, titled 'Robust Calibration for SVI Model Arbitrage Free,' available at SSRN 3543766 (2020)*.

### Parameters Boundaries and Arbitrage Constraints

We can summarize all the constraints that could guarantee to get SVI model with arbitrage free following these conditions

$$
\left\{\begin{array}{c}
0<a_{\min }=10^{-5} \leq a \leq \max \left(\omega^{mkt}\right) \\
0<b_{\min }=0.001<b<1 \\
-1<\rho<1\\
2 \min _i k_i \leq m \leq 2 \max _i k_i \\
0<\sigma_{\min }=0.01 \leq \sigma \leq \sigma_{\max }=1\\
g(k)>0\\
\partial_T \omega(k,T)\geq 0,\, \forall k\in\mathbb{R}\, ,T>0
\end{array}\right.
$$

We denote $x=k-m$, $R=\sqrt{x^2+\sigma^2}$, $W = \omega(k) = a+b\left(\rho x + R\right)$, $U = \omega^{\prime}(k) = b\left(\rho+\frac{x}{R}\right)$, $\omega^{\prime \prime}(k) = b\frac{\sigma^2}{R^3}$, then

$$
\begin{aligned}
    g(k)&=\left(1-\frac{k \omega^{\prime}(k)}{2 \omega(k)}\right)^2-\frac{\omega^{\prime}(k)^2}{4}\left(\frac{1}{\omega(k)}+\frac{1}{4}\right)+\frac{\omega^{\prime \prime}(k)}{2}\\
    &= \left(1-\frac{k U}{2 W}\right)^2-\frac{U^2}{4}\left(\frac{1}{W}+\frac{1}{4}\right)+\frac{b \sigma^2}{2 R^3}.
\end{aligned}
$$

### The Initial Guess

The initial guess values for the SVI calibration are summarized as follows
$$
\left\{\begin{array}{c}
a = \frac{1}{2}\min(\omega^{mkt}) \\
b=0.1 \\
\rho=-0.5\\
m = 0.1 \\
\sigma = 0.1
\end{array}\right.
$$

### Sequential Least-Squares Quadratic Programming (SLSQP) For SVI Calibration

We define the least-Squares objective function $f\left(k ; \chi_R\right)$ to optimize, where $\chi_R=\{a, b, \rho, m, \sigma\}$ is the set of the parameters model, for an expiry time fix $T$. $k=\left\{k_i\right\}_{i=1}^n$, where $k_i = \log \frac{K_i}{F_T}$.

$$
f\left(k ; \chi_R\right) = \sum_{i=1}^n \left[a+b\left\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\right\} - \omega_i^{mkt}\right]^2
$$

For a given parameter set $\chi_R=\{a, b, \rho, m, \sigma\}$, the *raw SVI parameterization* of total implied variance reads:

$$
w\left(k ; \chi_R\right)=a+b\left\{\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\right\}
$$

The problem reduced to find the optimal model’s parameters $\chi_R=\{a^*, b^*, \rho^*, m^*, \sigma^*\}$ by solving the following Non Linear Problem (NLP). We fix a constant $\epsilon>0$.

$$
\begin{array}{c}
& \min_{x \in \mathbb{R}^5} f(k; \chi_R) \\
\text{s.t.}& a_{\ell} \leq a \leq a_{u} \\
& b_{\ell} \leq b \leq b_{u} \\
& \rho_{\ell} \leq \rho \leq \rho_{u} \\
& m_{\ell} \leq m \leq m_{u} \\
& \sigma_{\ell} \leq \sigma \leq \sigma_{u} \\
& g(k; \chi_R) > \epsilon\\
& \frac{\partial \omega(k, T)}{\partial T} \geq \epsilon, \quad \forall k \in \mathbb{R}, \, T > 0
\end{array}
$$

The last condition is used in multi-slices SVI calibration to avoid calendar spread arbitrage. In practice, we start the calibration with SVI slice corresponding to the lowest maturity and more we move up to the next maturity more we add the constraint of non-crossing slices as bellow.

$$
\left\{\begin{array}{l}
\omega\left(k, T_0\right)>\epsilon \\
\omega\left(k, T_1\right)>\omega\left(k, T_0\right)+\epsilon \\
\vdots \\
\omega\left(k, T_i\right)>\omega\left(k, T_{i-1}\right)+\epsilon \quad 1 \leq i \leq n
\end{array}\right.
$$

Following this procedure, we can guarantee to get an SVI calibration with butterfly and calendar spread arbitrage free in the same time by running only one calibration for all the slices. Moreover, even if there is an arbitrage (butterfly or calendar spread) in our input data, our calibration can avoid these arbitrages and correct the model.

## Replication

For replication and practice, see **README** file for more details. Modifications are present there also.
