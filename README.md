# OLS Instability Under Multicollinearity and How Ridge Fixes It

## Overview

This project demonstrates mathematically why Ordinary Least Squares (OLS)
becomes unreliable when explanatory variables are correlated, and how
Ridge regression solves this problem.

All models are implemented from scratch using NumPy only. No machine
learning library is used for estimation.

## The Problem

The OLS analytical estimator is:

$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$

When two or more variables are correlated (multicollinearity), the matrix
$X^\top X$ becomes nearly singular. Its inversion becomes numerically
unstable, causing the estimated coefficients to diverge from their true
values.

## What This Project Shows

Using synthetic data with known true coefficients ($\beta_1 = 2$,
$\beta_2 = 3$), we vary the correlation $\rho$ between two explanatory
variables from 0 to 0.9999 and observe:

1. OLS coefficients remain stable for low $\rho$ but diverge completely as $\rho \to 1$
2. The condition number $\kappa(X^\top X) = \lambda_{max} / \lambda_{min}$ grows exponentially, confirming the numerical instability
3. Ridge regression stabilizes the estimates by shifting the eigenvalues of $X^\top X$
4. The ridge trace reveals the bias-variance tradeoff controlled by $\lambda$
5. Coefficient instability does not always translate into higher prediction error, which has important implications for interpretability

## Key Results

| $\rho$ | Condition number $\kappa$ | OLS $\hat{\beta}_1$ | OLS $\hat{\beta}_2$ |
|--------|--------------------------|---------------------|---------------------|
| 0.0    | 1.25                     | 1.991               | 2.973               |
| 0.5    | 3.53                     | 2.032               | 2.978               |
| 0.9    | 22.24                    | 2.064               | 2.945               |
| 0.99   | 232.85                   | 2.193               | 2.816               |
| 0.999  | 2338.96                  | 2.600               | 2.408               |
| 0.9999 | 23400.00                 | 3.889               | 1.120               |

True values: $\beta_1 = 2.0$, $\beta_2 = 3.0$

## Project Structure

```
ols-instability-ridge-regularization/
│
├── notebook.ipynb
├── README.md
└── images/
    ├── ols_instability_coeffs.png
    ├── condition_number.png
    ├── ols_vs_ridge.png
    ├── ridge_trace.png
    └── mse_comparison.png
```

## Concepts Covered

- OLS analytical estimator and its matrix formulation
- Multicollinearity and its effect on $X^\top X$
- Condition number as a rigorous diagnostic tool
- Ridge analytical estimator and L2 regularization
- Eigenvalue interpretation of Ridge stability
- Bias-variance tradeoff controlled by $\lambda$
- Ridge trace for visual hyperparameter selection
- Distinction between coefficient stability and prediction error

## Requirements

```
numpy
pandas
matplotlib
seaborn
```

## Usage

```bash
git clone https://github.com/capristunt/ols-instability-ridge-regularization
cd ols-instability-ridge-regularization
pip install -r requirements.txt
jupyter notebook
```