# SAEnet: Structure-Adaptive Elastic Net

<img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">

## Overview

SAEnet implements the Structure-Adaptive Elastic Net method for high-dimensional regression with external structural information. This package extends the elastic net framework by incorporating external information about predictors to improve variable selection and prediction performance.

## Installation

You can install the development version of SAEnet from GitHub:

```r
# Install devtools if you haven't already
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}

# Install SAEnet from GitHub
devtools::install_github("zhangxiany-tamu/SAEnet")
```

## Key Features

- **Structure-based penalization**: Incorporates external structural information (group-based or covariate-dependent) to improve variable selection
- **Adaptive weights**: Uses an iterative procedure to refine penalty weights based on coefficient estimates
- **Multiple selection criteria**: Supports both cross-validation ("lambda.min", "lambda.1se") and BIC for model selection
- **Visualization tools**: Built-in methods for model visualization and coefficient plotting
- **Parallel processing**: Support for parallel computation to speed up cross-validation

## Demonstrating the Benefit of Structural Information

```r
library(SAEnet)
set.seed(123)
n <- 100       # Number of samples
p <- 500       # Number of variables

# Generate covariate that influences variable importance
# Create a covariate that will determine the probability of non-zero coefficients
z_covariate <- runif(p, 0, 1)  # Uniformly distributed between 0 and 1

# Define a function that maps the covariate to probability of being non-zero
# Higher z values correspond to higher probability of signal
prob_nonzero <- function(z) {
  pmin(0.5, z^2)  # Quadratic relationship capped at 0.8
}

# Generate true model coefficients using Bernoulli trials
# Variables with higher z_covariate values are more likely to be non-zero
is_nonzero <- rbinom(p, 1, prob_nonzero(z_covariate))
true_coef <- rep(0, p)

# For coefficients that are non-zero, generate effect sizes
# Effect sizes proportional to z_covariate for non-zero coefficients
effect_size <- function(z) {
  sign(rnorm(1)) * (0.5 + 1.5 * z)  # Mixed signs, magnitude related to z
}

for (j in 1:p) {
  if (is_nonzero[j] == 1) {
    true_coef[j] <- effect_size(z_covariate[j])
  }
}

# Generate design matrix
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", 1:p)

# Generate response with moderate noise
y <- X %*% true_coef + rnorm(n, 0, 1.0)

# Create covariate structure information for SAEnet
# This is where we explicitly inform the model about the covariate
structure_info_covariate <- list(covariate = cbind(z_covariate,z_covariate^2))

# Fit models for comparison
# 1. Standard Elastic Net (no structure)
fit_std <- saenet(y = y, x = X, 
                 lambda_selection_rule = "bic", 
                 max_iterations = 5)

# 2. SAEnet with covariate structure
fit_covariate <- saenet(y = y, x = X, 
                        lambda_selection_rule = "bic",
                        structure_info = structure_info_covariate,
                        max_iterations = 5)

# Compare results
bic_std <- tail(fit_std$criterion_value, 1)
bic_covariate <- tail(fit_covariate$criterion_value, 1)

# Get coefficient estimates
beta_std <- predict(fit_std, type = "coefficients")
beta_covariate <- predict(fit_covariate, type = "coefficients")

# Calculate performance metrics
# True positives: non-zero coefficients correctly identified
tp_std <- sum(abs(beta_std) > 1e-8 & true_coef != 0)
tp_covariate <- sum(abs(beta_covariate) > 1e-8 & true_coef != 0)

# False positives: zero coefficients incorrectly identified as non-zero
fp_std <- sum(abs(beta_std) > 1e-8 & true_coef == 0)
fp_covariate <- sum(abs(beta_covariate) > 1e-8 & true_coef == 0)

# Display results
cat(sprintf("Number of true non-zero coefficients: %d (%.1f%%)\n", 
            sum(true_coef != 0), 100*sum(true_coef != 0)/p))

cat(sprintf("\nStandard Elastic Net:\n"))
cat(sprintf("  BIC = %.2f\n", bic_std))
cat(sprintf("  True Positives: %d/%d (%.1f%%)\n", 
            tp_std, sum(true_coef != 0), 100*tp_std/sum(true_coef != 0)))
cat(sprintf("  False Positives: %d (%.1f%%)\n", 
            fp_std, 100*fp_std/sum(true_coef == 0)))

cat(sprintf("\nCovariate-Based SAEnet:\n"))
cat(sprintf("  BIC = %.2f\n", bic_covariate))
cat(sprintf("  True Positives: %d/%d (%.1f%%)\n", 
            tp_covariate, sum(true_coef != 0), 100*tp_covariate/sum(true_coef != 0)))
cat(sprintf("  False Positives: %d (%.1f%%)\n", 
            fp_covariate, 100*fp_covariate/sum(true_coef == 0)))

# The key advantages demonstrated:
# 1. SAEnet with covariate information achieves better overall performance than 
#    standard elastic net when the covariate truly relates to variable importance
# 2. The covariate-based approach significantly improves true positive rates,
#    especially in higher-value covariate ranges where signals are more likely
# 3. The covariate-based approach maintains similar control of false positives
```

```
# Simulation example with three groups having distinct sparsity levels
set.seed(456)
n <- 120       # Number of samples
p_group1 <- 50 # Variables in group 1
p_group2 <- 300 # Variables in group 2
p_group3 <- 40 # Variables in group 3
p <- p_group1 + p_group2 + p_group3  # Total variables (120)

# Create group structure
group_membership <- c(rep(1, p_group1), rep(2, p_group2), rep(3, p_group3))

# Define sparsity levels for each group
sparsity_group1 <- 0.4  # 40% of variables in group 1 are active
sparsity_group2 <- 0.0  # No active variables in group 2
sparsity_group3 <- 0.9  # 90% of variables in group 3 are active

# Generate X matrix with correlations within groups
X <- matrix(0, n, p)
colnames(X) <- paste0("X", 1:p)

# Generate data with correlation within groups
# Group-specific correlation matrices
rho1 <- 0.3  # Moderate correlation in group 1
rho2 <- 0.5  # Stronger correlation in group 2
rho3 <- 0.4  # Moderate correlation in group 3

# Function to generate correlated data for a group
generate_group_data <- function(n_samples, n_vars, rho) {
  if (n_vars == 1) return(matrix(rnorm(n_samples), ncol = 1))
  
  # Create correlation matrix
  cor_matrix <- matrix(rho, nrow = n_vars, ncol = n_vars)
  diag(cor_matrix) <- 1
  
  # Cholesky decomposition
  chol_matrix <- chol(cor_matrix)
  
  # Generate data
  Z <- matrix(rnorm(n_samples * n_vars), nrow = n_samples)
  return(Z %*% chol_matrix)
}

# Generate group-specific data
X[, group_membership == 1] <- generate_group_data(n, p_group1, rho1)
X[, group_membership == 2] <- generate_group_data(n, p_group2, rho2)
X[, group_membership == 3] <- generate_group_data(n, p_group3, rho3)

# Generate true coefficients with specified sparsity patterns
beta_true <- rep(0, p)

# Uniform signal strength for all active variables
signal_strength <- 0.3

# Group 1: 40% active
active_indices_g1 <- sample(which(group_membership == 1), 
                            round(p_group1 * sparsity_group1))
beta_true[active_indices_g1] <- signal_strength

# Group 2: All zeros (0% active)
# No action needed

# Group 3: 90% active
active_indices_g3 <- sample(which(group_membership == 3), 
                            round(p_group3 * sparsity_group3))
beta_true[active_indices_g3] <- signal_strength

# Generate response with moderate noise
sigma <- 1.2  # Noise level
y <- X %*% beta_true + rnorm(n, 0, sigma)

# Create structure information for SAEnet
structure_info <- list(group = group_membership)

# Fit models for comparison
# 1. Standard Elastic Net (no group structure)
fit_std <- saenet(y = y, x = X, 
                 lambda_selection_rule = "bic", 
                 max_iterations = 5)

# 2. SAEnet with group structure
fit_group <- saenet(y = y, x = X, 
                   lambda_selection_rule = "bic",
                   structure_info = structure_info,
                   max_iterations = 5)

# Compare overall results
# Get coefficient estimates
beta_std <- predict(fit_std, type = "coefficients")
beta_group <- predict(fit_group, type = "coefficients")

# Calculate performance metrics
tp_std <- sum(abs(beta_std) > 1e-8 & beta_true != 0)
tp_group <- sum(abs(beta_group) > 1e-8 & beta_true != 0)

fp_std <- sum(abs(beta_std) > 1e-8 & beta_true == 0)
fp_group <- sum(abs(beta_group) > 1e-8 & beta_true == 0)

# Display overall results
cat("Overall Results:\n")
cat(sprintf("True model: %d non-zero coefficients (%.1f%%)\n", 
            sum(beta_true != 0), 100*sum(beta_true != 0)/p))

cat(sprintf("\nStandard Elastic Net:\n"))
cat(sprintf("  BIC = %.2f\n", tail(fit_std$criterion_value, 1)))
cat(sprintf("  True Positives: %d/%d (%.1f%%)\n", 
            tp_std, sum(beta_true != 0), 100*tp_std/sum(beta_true != 0)))
cat(sprintf("  False Positives: %d/%d (%.1f%%)\n", 
            fp_std, sum(beta_true == 0), 100*fp_std/sum(beta_true == 0)))

cat(sprintf("\nSAEnet with Group Structure:\n"))
cat(sprintf("  BIC = %.2f\n", tail(fit_group$criterion_value, 1)))
cat(sprintf("  True Positives: %d/%d (%.1f%%)\n", 
            tp_group, sum(beta_true != 0), 100*tp_group/sum(beta_true != 0)))
cat(sprintf("  False Positives: %d/%d (%.1f%%)\n", 
            fp_group, sum(beta_true == 0), 100*fp_group/sum(beta_true == 0)))

# Group-level analysis
group_analysis <- data.frame(
  Group = c(1, 2, 3),
  Size = c(p_group1, p_group2, p_group3),
  TrueSparsity = c(sparsity_group1, sparsity_group2, sparsity_group3) * 100,
  TrueNonZero = c(
    sum(beta_true[group_membership == 1] != 0),
    sum(beta_true[group_membership == 2] != 0),
    sum(beta_true[group_membership == 3] != 0)
  )
)

# Calculate TP and FP rates by group
for (g in 1:3) {
  group_indices <- which(group_membership == g)
  true_nonzero <- beta_true[group_indices] != 0
  true_zero <- beta_true[group_indices] == 0
  
  # Standard model
  std_nonzero <- abs(beta_std[group_indices]) > 1e-8
  
  # Group model
  group_nonzero <- abs(beta_group[group_indices]) > 1e-8
  
  # True positive rates
  if (sum(true_nonzero) > 0) {
    group_analysis$StdTPRate[g] <- 100 * sum(std_nonzero & true_nonzero) / sum(true_nonzero)
    group_analysis$GroupTPRate[g] <- 100 * sum(group_nonzero & true_nonzero) / sum(true_nonzero)
  } else {
    group_analysis$StdTPRate[g] <- NA
    group_analysis$GroupTPRate[g] <- NA
  }
  
  # False positive rates
  if (sum(true_zero) > 0) {
    group_analysis$StdFPRate[g] <- 100 * sum(std_nonzero & true_zero) / sum(true_zero)
    group_analysis$GroupFPRate[g] <- 100 * sum(group_nonzero & true_zero) / sum(true_zero)
  } else {
    group_analysis$StdFPRate[g] <- NA
    group_analysis$GroupFPRate[g] <- NA
  }
}

# Display group-level results
cat("\nGroup-Level Analysis:\n")
print(group_analysis[, c("Group", "Size", "TrueSparsity", "TrueNonZero", 
                         "StdTPRate", "GroupTPRate", "StdFPRate", "GroupFPRate")])
```

### Comparing Selection Criteria: BIC vs Cross-Validation

```r
# Generate data with a sparse true beta
set.seed(456)
n <- 120  # number of observations
p <- 100  # number of variables
X_select <- matrix(rnorm(n * p), n, p)
colnames(X_select) <- paste0("X", 1:p)

# True beta with 7 non-zero coefficients
true_beta_select <- rep(0, p)
true_beta_select[c(5, 10, 15, 20, 25, 30, 35)] <- c(1.2, 1.0, 0.8, -1.5, -1.0, -0.7, 0.5)
y_select <- X_select %*% true_beta_select + rnorm(n, 0, 0.5)

# 1. Use BIC for model selection
fit_bic <- saenet(y = y_select, x = X_select, 
                 lambda_selection_rule = "bic",
                 max_iterations = 3)

# 2. Use CV with minimum error rule
fit_cv_min <- saenet(y = y_select, x = X_select, 
                    lambda_selection_rule = "lambda.min", 
                    max_iterations = 3, 
                    num_folds = 5)

# 3. Use CV with 1SE rule (more parsimonious)
fit_cv_1se <- saenet(y = y_select, x = X_select, 
                    lambda_selection_rule = "lambda.1se", 
                    max_iterations = 3, 
                    num_folds = 5)

# Compare number of non-zero coefficients
nz_bic <- sum(abs(predict(fit_bic, type = "coefficients")) > 1e-8)
nz_cv_min <- sum(abs(predict(fit_cv_min, type = "coefficients")) > 1e-8)
nz_cv_1se <- sum(abs(predict(fit_cv_1se, type = "coefficients")) > 1e-8)

cat(sprintf("True model has 7 non-zero coefficients\n"))
cat(sprintf("BIC selection: %d non-zero coefficients\n", nz_bic))
cat(sprintf("CV min selection: %d non-zero coefficients\n", nz_cv_min))
cat(sprintf("CV 1SE selection: %d non-zero coefficients\n", nz_cv_1se))

# BIC often gives more parsimonious models than lambda.min and
# may be closer to the true model size when n is sufficiently large
# The 1SE rule typically gives more parsimonious models than lambda.min
```

## Visualization

SAEnet provides built-in plotting functions:

```r
# Plot coefficients from the final iteration
plot(fit, type = "coefficients")

# Show only top 10 variables by absolute coefficient value
plot(fit, type = "coefficients", top_n = 20)

# Plot coefficients from a specific iteration
plot(fit, type = "coefficients", iteration = 2)
```

## Technical Details

The Structure-Adaptive Elastic Net algorithm follows an iterative procedure:

1. Initialize by fitting a standard elastic net
2. Generate adaptive weights based on the structural information
3. Solve the weighted elastic net problem
4. Repeat steps 2-3 for a specified number of iterations

The method incorporates three different approaches to weight generation:

- **Standard adaptive weights**: When no structural information is provided, weights are calculated as $1/|\hat{\beta}|^\gamma$ for each coefficient
- **Group-based structure**: Group-level information is used to assign identical weights to variables within the same group, based on the mean absolute coefficient value within each group
- **Covariate-dependent structure**: A nonlinear optimization is performed to find the optimal relationship between covariates and penalty weights

The package supports both cross-validation (with "lambda.min" or "lambda.1se" options) and BIC for model selection, allowing users to choose between prediction accuracy and model parsimony.

## Citation

If you use SAEnet in your research, please cite:

```
Pramanik, S., & Zhang, X. (2020). Structure Adaptive Elastic-Net. arXiv:2006.02041.
```

## License

This package is released under the MIT License.
