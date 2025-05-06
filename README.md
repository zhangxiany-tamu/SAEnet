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
devtools::install_github("username/SAEnet")
```

## Key Features

- **Structure-based penalization**: Incorporates external structural information (group-based or covariate-dependent) to improve variable selection
- **Adaptive weights**: Uses an iterative procedure to refine penalty weights based on coefficient estimates
- **Cross-validation**: Automatically tunes all hyperparameters via cross-validation
- **Visualization tools**: Built-in methods for model visualization and coefficient plotting
- **Parallel processing**: Support for parallel computation to speed up cross-validation

## Basic Usage

```r
library(SAEnet)

# Example: Standard Adaptive Elastic Net (No Structure)
set.seed(123)
n <- 100  # number of observations
p <- 200  # number of variables
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(rep(1, 10), rep(0, p - 10))
y <- X %*% beta_true + rnorm(n, 0, 0.5)

# Fit the model
fit <- saenet(y = y, x = X, max_iterations = 3, num_folds = 5)

# View model summary
print(fit)

# Plot non-zero coefficients
plot(fit, type = "coefficients")

# Plot CV error across iterations
plot(fit, type = "cv.error")

# Make predictions with new data
predictions <- predict(fit, newx = X)
```

## Advanced Usage: Incorporating Structural Information

### Group-Based Structure

```r
# Create arbitrary groups for variables (10 groups of 20 variables each)
groups <- rep(1:10, each = 20)
structure_info <- list(group = groups)

# Fit the model with group structure
fit_group <- saenet(y = y, x = X, structure_info = structure_info,
                   max_iterations = 3, num_folds = 5)
```

### Covariate-Dependent Structure

```r
# Create arbitrary covariate related to variable importance
covariate <- runif(p)
covariate[1:10] <- covariate[1:10] + 0.5  # Make important variables have higher values
structure_info <- list(covariate = matrix(covariate, ncol = 1))

# Fit the model with covariate structure
fit_covariate <- saenet(y = y, x = X, structure_info = structure_info,
                       max_iterations = 3, num_folds = 5)
```

## Additional Options

```r
# Customize optimization parameters
fit_custom <- saenet(
  y = y, x = X,
  max_iterations = 10,                          # More iterations for convergence
  standardize = TRUE,                           # Standardize predictors
  alpha_sequence = seq(0, 1, by = 0.05),        # Finer grid for alpha
  gamma_sequence = c(0.5, 0.7, 1.0),            # Custom gamma values
  num_folds = 10,                               # More folds for CV
  num_cores = 2                                 # Parallel processing
)
```

## Visualization

SAEnet provides built-in plotting functions:

```r
# Plot coefficients from the final iteration
plot(fit, type = "coefficients")

# Show only top 10 variables by absolute coefficient value
plot(fit, type = "coefficients", top_n = 10)

# Plot coefficients from a specific iteration
plot(fit, type = "coefficients", iteration = 2)

# Plot cross-validation error across iterations
plot(fit, type = "cv.error")
```

## Citation

If you use SAEnet in your research, please cite:

```
Pramanik, S., & Zhang, X. (2020). Structure Adaptive Elastic-Net. arXiv:2006.02041.
```

## License

This package is released under the MIT License.
