#' Structure-Adaptive Elastic Net
#'
#' @description
#' Implements the Structure-Adaptive Elastic Net (SAEnet) method, which extends
#' the elastic net framework by incorporating external structural information.
#' This approach translates external information about predictors into different
#' penalization strengths for the regression coefficients.
#'
#' @details
#' The Structure-Adaptive Elastic Net incorporates external structural information
#' to improve variable selection in high-dimensional regression. The method adapts
#' the standard elastic net by allowing the adaptive weights to depend on external
#' structural information, which can be either group-based or covariate-dependent.
#'
#' The algorithm follows an iterative procedure:
#' 1. Initialize by fitting a standard elastic net
#' 2. Generate adaptive weights based on the structural information
#' 3. Solve the weighted elastic net problem
#' 4. Repeat steps 2-3 for a specified number of iterations
#'
#' @param formula An object of class formula: a symbolic description of the model to be fitted
#' @param data A data frame containing the variables in the model
#' @param x Design matrix of predictors (n × p) (alternative to formula/data)
#' @param y Response vector (alternative to formula/data)
#' @param structure_info External structural information. Can be:
#'   \itemize{
#'     \item NULL: No structural information (standard adaptive elastic net)
#'     \item List with `group`: Group-based structural info with `$group` component
#'           indicating group membership
#'     \item List with `covariate`: Covariate-dependent structural info with `$covariate`
#'           component containing covariate values
#'   }
#' @param max_iterations Number of iterations for the adaptive penalty procedure (default: 5)
#' @param standardize Logical. Whether to standardize the predictors (default: TRUE)
#' @param alpha_sequence A sequence of elastic net mixing parameters (0 <= alpha <= 1) to try,
#'        where alpha=1 is lasso and alpha=0 is ridge (default: seq(0, 1, by = 0.1))
#' @param nlambda Number of lambda values to generate in the sequence (default: 100)
#' @param lambda_min_ratio Ratio of smallest to largest lambda value in the sequence
#'        (default: 1e-4 for n >= p, 0.01 for n < p)
#' @param gamma_sequence A sequence of power parameters for adaptive weights
#'        (default: seq(0.1, 1, by = 0.1))
#' @param num_folds Number of folds for cross-validation (default: 10)
#' @param num_cores Number of cores for parallel computation (default: 1)
#' @param verbose Logical. Whether to print progress messages (default: TRUE)
#'
#' @return An object of class "saenet" with the following components:
#'   \itemize{
#'     \item intercept: Vector of estimated intercept values for each iteration
#'     \item coefficients: Matrix of estimated coefficient vectors (on original scale)
#'          with columns corresponding to iterations
#'     \item weights: Matrix of adaptive weights for each iteration
#'     \item cv_errors: Vector of cross-validation errors for each iteration
#'     \item lambda: Vector of selected lambda parameters for each iteration
#'     \item alpha: Vector of selected alpha parameters for each iteration
#'     \item gamma: Vector of selected gamma parameters for each iteration
#'     \item num_iterations: Number of iterations performed
#'     \item feature_scaling: Scaling information for predictors (if standardize=TRUE)
#'   }
#'
#' @references
#' Pramanik, S., & Zhang, X. (2020). Structure Adaptive Elastic-Net. arXiv:2006.02041.
#'
#' @examples
#' # Example 1: Standard Adaptive Elastic Net (No Structure)
#' set.seed(123)
#' n <- 100  # number of observations
#' p <- 200  # number of variables
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(rep(1, 10), rep(0, p - 10))
#' y <- X %*% beta_true + rnorm(n, 0, 0.5)
#'
#' # Fit the model
#' fit <- saenet(y = y, x = X, max_iterations = 3, num_folds = 5)
#'
#' # View model summary
#' print(fit)
#'
#' # Extract coefficients from the final iteration
#' beta_est <- predict(fit, type = "coefficients")
#'
#' # Make predictions for new data
#' predictions <- predict(fit, newx = X)
#'
#' # Plot non-zero coefficients
#' plot(fit, type = "coefficients")
#'
#' # Plot CV error across iterations
#' plot(fit, type = "cv.error")
#'
#' # Example 2: Group-based Structure
#' # Create arbitrary groups for variables (10 groups of 20 variables each)
#' groups <- rep(1:10, each = 20)
#' structure_info <- list(group = groups)
#'
#' # Fit the model with group structure
#' fit_group <- saenet(y = y, x = X, structure_info = structure_info,
#'                     max_iterations = 3, num_folds = 5)
#'
#' # Plot non-zero coefficients
#' plot(fit_group, type = "coefficients")
#'
#' # Example 3: Covariate-dependent Structure
#' # Create arbitrary covariate related to variable importance
#' # (higher values for more important variables)
#' covariate <- runif(p)
#' covariate[1:10] <- covariate[1:10] + 0.5  # make important variables have higher values
#' structure_info <- list(covariate = matrix(covariate, ncol = 1))
#'
#' # Fit the model with covariate structure
#' fit_covariate <- saenet(y = y, x = X, structure_info = structure_info,
#'                         max_iterations = 3, num_folds = 5)
#'
#' # Plot non-zero coefficients
#' plot(fit_covariate, type = "coefficients")
#'
#' @export
saenet <- function(formula, data, x, y, structure_info, max_iterations = 5,
                   standardize = TRUE, alpha_sequence = seq(0, 1, by = 0.1), nlambda = 100,
                   lambda_min_ratio = ifelse(nobs < nvars, 0.01, 1e-04), gamma_sequence = seq(0.1, 1, by = 0.1),
                   num_folds = 10, num_cores = 1, verbose = TRUE) {

  # Load required packages
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package glmnet is required but not installed")
  }
  if (!requireNamespace("foreach", quietly = TRUE)) {
    stop("Package foreach is required but not installed")
  }
  if (!requireNamespace("doParallel", quietly = TRUE)) {
    stop("Package doParallel is required but not installed")
  }

  # Check if formula or x,y interface is used
  if (!missing(formula)) {
    # Formula interface
    if (missing(data)) {
      data <- environment(formula)
    }
    model_frame <- model.frame(formula, data)
    y <- model.response(model_frame, "numeric")
    X <- model.matrix(formula, model_frame)[, -1, drop = FALSE]  # Remove intercept column
  } else if (!missing(x) && !missing(y)) {
    # x,y interface
    X <- as.matrix(x)
    y <- as.numeric(y)
  } else {
    stop("Either 'formula' and 'data' or 'x' and 'y' must be provided")
  }

  # Validate input dimensions
  n_rows_X <- nrow(X)
  n_length_y <- length(y)

  if (n_rows_X != n_length_y) {
    stop("Number of observations from y and X should match. The length of y and the number of rows in X do not match!")
  } else {
    nobs <- n_length_y
  }

  # Handle optional parameters
  if (missing(num_cores)) num_cores <- 1

  # Number of predictor variables
  nvars <- ncol(X)

  # Standardize X if requested
  if (standardize) {
    feature_means <- colMeans(X)
    feature_sds <- apply(X, 2, sd)
    feature_sds[feature_sds == 0] <- 1  # Protect against division by zero
    X_standardized <- scale(X, center = feature_means, scale = feature_sds)
  } else {
    X_standardized <- X
  }

  # Initialize storage for outputs
  coef_std_matrix <- weights_matrix <- matrix(nrow = nvars, ncol = 1 + max_iterations)
  intercept_std_vector <- cv_error_vector <- lambda_vector <- alpha_vector <- gamma_vector <- rep(NA, 1 + max_iterations)

  # Use the unified fitting function
  fit_results <- fit_adaptive_net(
    y = y,
    X_std = X_standardized,
    structure_info = if(missing(structure_info)) NULL else structure_info,
    max_iterations = max_iterations,
    standardize_X = standardize,
    alpha_sequence = alpha_sequence,
    nlambda = nlambda,
    lambda_min_ratio = lambda_min_ratio,
    gamma_sequence = gamma_sequence,
    num_folds = num_folds,
    num_cores = num_cores,
    verbose = verbose,
    nvars = nvars
  )

  # Extract results
  coef_std_matrix <- fit_results$coef_std_matrix
  intercept_std_vector <- fit_results$intercept_std_vector
  weights_matrix <- fit_results$weights_matrix
  cv_error_vector <- fit_results$cv_error_vector
  lambda_vector <- fit_results$lambda_vector
  alpha_vector <- fit_results$alpha_vector
  gamma_vector <- fit_results$gamma_vector

  # Transform coefficients back to original scale if standardized
  if (standardize) {
    # Back-transformation formula: beta_orig = beta_std / scale
    coef_matrix <- matrix(NA, nrow = nvars, ncol = ncol(coef_std_matrix))
    for (j in 1:ncol(coef_std_matrix)) {
      coef_matrix[, j] <- coef_std_matrix[, j] / feature_sds
      intercept_std_vector[j] <- intercept_std_vector[j] - sum(feature_means * coef_matrix[, j])
    }
  } else {
    coef_matrix <- coef_std_matrix
  }

  # Add variable names to coefficients if available
  if (!is.null(colnames(X))) {
    rownames(coef_matrix) <- colnames(X)
    rownames(coef_std_matrix) <- colnames(X)
  }

  # Construct and return the result
  result <- list(
    'intercept' = intercept_std_vector,
    'coefficients' = as.matrix(coef_matrix),
    'weights' = weights_matrix,
    'cv_errors' = cv_error_vector,
    'lambda' = lambda_vector,
    'alpha' = alpha_vector,
    'gamma' = gamma_vector,
    'num_iterations' = max_iterations
  )

  # Add scaling information if standardization was used
  if (standardize) {
    result$feature_scaling <- list(
      center = feature_means,
      scale = feature_sds
    )
  }

  # Add additional class information
  class(result) <- c("saenet", "list")

  return(result)
}

#' Fit Structure-Adaptive Elastic Net model
#'
#' @description
#' Internal function that implements the core fitting algorithm for SAEnet models.
#'
#' @param y Response vector
#' @param X_std Standardized design matrix
#' @param structure_info External structural information (NULL, group-based, or covariate-based)
#' @param max_iterations Maximum number of iterations
#' @param standardize_X Whether to standardize the predictors
#' @param alpha_sequence Sequence of alpha values to try
#' @param nlambda Number of lambda values
#' @param lambda_min_ratio Ratio of min to max lambda
#' @param gamma_sequence Sequence of gamma values to try
#' @param num_folds Number of cross-validation folds
#' @param num_cores Number of cores for parallel computation
#' @param verbose Whether to print progress messages
#' @param nvars Number of predictor variables
#'
#' @return A list with the following components:
#'   \itemize{
#'     \item coef_std_matrix: Matrix of coefficient estimates (standardized scale)
#'     \item intercept_std_vector: Vector of intercept estimates
#'     \item weights_matrix: Matrix of adaptive weights
#'     \item cv_error_vector: Vector of cross-validation errors
#'     \item lambda_vector: Vector of selected lambda values
#'     \item alpha_vector: Vector of selected alpha values
#'     \item gamma_vector: Vector of selected gamma values
#'   }
#'
#' @keywords internal
fit_adaptive_net <- function(y, X_std, structure_info, max_iterations, standardize_X,
                             alpha_sequence, nlambda, lambda_min_ratio, gamma_sequence, num_folds,
                             num_cores, verbose, nvars) {

  # Determine the structure type
  if (is.null(structure_info)) {
    structure_type <- "standard"
  } else if (!is.null(names(structure_info)) && names(structure_info)[1] == 'group') {
    structure_type <- "group"
    # Extract group information
    group_ids <- unique(structure_info$group)
    num_groups <- length(group_ids)
    group_indices <- vector(mode = 'list', length = num_groups)
    for (group_idx in 1:num_groups) {
      group_indices[[group_idx]] <- which(structure_info$group == group_ids[group_idx])
    }
  } else if (!is.null(names(structure_info)) && names(structure_info)[1] == 'covariate') {
    structure_type <- "covariate"
    # Extract and standardize covariate
    covariate <- structure_info$covariate
    if (!is.matrix(covariate) && !is.data.frame(covariate)) {
      covariate <- as.matrix(covariate, ncol = 1)
    }
    covariate_means <- colMeans(covariate)
    covariate_sds <- apply(covariate, 2, sd)
    covariate_std <- scale(covariate, center = covariate_means, scale = covariate_sds)
  } else {
    stop("Invalid structure_info. Must be NULL or a list with 'group' or 'covariate' element.")
  }

  # Storage for outputs
  coef_std_matrix <- weights_matrix <- matrix(nrow = nvars, ncol = 1 + max_iterations)
  intercept_std_vector <- cv_error_vector <- lambda_vector <- alpha_vector <- gamma_vector <- rep(NA, 1 + max_iterations)

  # Register parallel backend
  doParallel::registerDoParallel(cores = num_cores)

  # Implementing Adaptive Enet with adaptive L1 and non-adaptive L2 penalties
  for (iteration in 0:max_iterations) {
    if (iteration == 0) {
      # Iteration 0 - Initial fit (same for all structure types)
      # Perform cross-validation over alpha sequence
      results <- cv_alpha(X_std, y, standardize_X, alpha_sequence,
                          nlambda, lambda_min_ratio, num_folds, num_cores)

      # Extract the best alpha, lambda and corresponding coefficients
      best_alpha <- results$best_alpha
      best_lambda <- results$best_lambda

      # Fit model with optimal alpha and lambda
      model_fit <- glmnet::glmnet(
        x = X_std, y = y,
        alpha = best_alpha,
        standardize = standardize_X,
        lambda = best_lambda
      )

      intercept_std_vector[iteration + 1] <- as.numeric(model_fit$a0)
      coef_std_matrix[, iteration + 1] <- as.numeric(as.matrix(model_fit$beta))
      cv_error_vector[iteration + 1] <- results$best_cv_error
      lambda_vector[iteration + 1] <- best_lambda
      alpha_vector[iteration + 1] <- best_alpha

    } else {
      # Iterations 1 to max_iterations - Structure-specific adaptive penalties

      # Calculate initial weights based on structure type
      if (structure_type == "standard") {
        initial_weights <- calculate_standard_weights(coef_std_matrix[, iteration])
      } else if (structure_type == "group") {
        initial_weights <- calculate_group_weights(coef_std_matrix[, iteration], group_indices, num_groups)
      } else if (structure_type == "covariate") {
        # Handle covariate structure
        cv_results <- cv_covariate(X_std, y, standardize_X, alpha_sequence,
                                   nlambda, lambda_min_ratio, gamma_sequence,
                                   coef_std_matrix[, iteration],
                                   covariate_std, nvars, num_folds,
                                   num_cores)

        optimal_weights <- pmin(exp(cv_results$optimal_tau_params[1] +
                                      covariate_std%*%cv_results$optimal_tau_params[-1]),
                                1e+100)

        # Fit model with optimal weights
        model_fit <- glmnet::glmnet(
          x = X_std, y = y,
          alpha = cv_results$best_alpha,
          standardize = standardize_X,
          penalty.factor = optimal_weights,
          lambda = cv_results$best_lambda
        )

        intercept_std_vector[iteration + 1] <- as.numeric(model_fit$a0)
        coef_std_matrix[, iteration + 1] <- as.numeric(as.matrix(model_fit$beta))
        weights_matrix[, iteration + 1] <- optimal_weights
        cv_error_vector[iteration + 1] <- cv_results$best_cv_error
        lambda_vector[iteration + 1] <- cv_results$best_lambda
        gamma_vector[iteration + 1] <- cv_results$best_gamma
        alpha_vector[iteration + 1] <- cv_results$best_alpha

        # Continue to next iteration
        if (verbose & iteration > 0) print(paste('Iteration', iteration, 'out of', max_iterations))
        next
      }

      # Handle cases where coefficients are all zero or weights are all infinite
      if (any(sum(coef_std_matrix[, iteration] != 0) == 0, sum(initial_weights != Inf) == 0)) {
        coef_std_matrix[, iteration + 1] <- numeric(nvars)
      } else {
        # Cross-validation to find optimal parameters (standard and group structures)
        cv_results <- cv_gamma(X_std, y, standardize_X, alpha_sequence,
                               nlambda, lambda_min_ratio, gamma_sequence,
                               initial_weights, num_folds, num_cores)

        optimal_weights <- pmin(initial_weights^cv_results$best_gamma, 1e+100)

        # Fit model with optimal weights
        model_fit <- glmnet::glmnet(
          x = X_std, y = y,
          alpha = cv_results$best_alpha,
          standardize = standardize_X,
          penalty.factor = optimal_weights,
          lambda = cv_results$best_lambda,
          nlambda = 1
        )

        intercept_std_vector[iteration + 1] <- as.numeric(model_fit$a0)
        coef_std_matrix[, iteration + 1] <- as.numeric(as.matrix(model_fit$beta))
        weights_matrix[, iteration + 1] <- optimal_weights
        cv_error_vector[iteration + 1] <- cv_results$best_cv_error
        lambda_vector[iteration + 1] <- cv_results$best_lambda
        gamma_vector[iteration + 1] <- cv_results$best_gamma
        alpha_vector[iteration + 1] <- cv_results$best_alpha
      }
    }

    if (verbose & iteration > 0) print(paste('Iteration', iteration, 'out of', max_iterations))
  }

  return(list(
    coef_std_matrix = coef_std_matrix,
    intercept_std_vector = intercept_std_vector,
    weights_matrix = weights_matrix,
    cv_error_vector = cv_error_vector,
    lambda_vector = lambda_vector,
    alpha_vector = alpha_vector,
    gamma_vector = gamma_vector
  ))
}

#' Calculate weights for standard adaptive elastic net
#'
#' @param coefficients Current coefficient estimates
#'
#' @return Vector of weights (1/|beta|)
#'
#' @keywords internal
calculate_standard_weights <- function(coefficients) {
  return(1 / abs(coefficients))
}

#' Calculate weights for group-based adaptive elastic net
#'
#' @param coefficients Current coefficient estimates
#' @param group_indices List of indices for each group
#' @param num_groups Number of groups
#'
#' @return Vector of weights (1/mean(|beta|) within each group)
#'
#' @keywords internal
calculate_group_weights <- function(coefficients, group_indices, num_groups) {
  weights <- rep(NA, length(coefficients))
  for (group_idx in 1:num_groups) {
    group_coefs <- coefficients[group_indices[[group_idx]]]
    weights[group_indices[[group_idx]]] <- 1 / mean(abs(group_coefs))
  }
  return(weights)
}

#' Cross-validation across alpha values
#'
#' @param X_std Standardized design matrix
#' @param y Response vector
#' @param standardize_X Whether to standardize the predictors
#' @param alpha_sequence Sequence of alpha values to try
#' @param nlambda Number of lambda values
#' @param lambda_min_ratio Ratio of min to max lambda
#' @param num_folds Number of cross-validation folds
#' @param num_cores Number of cores for parallel computation
#' @param penalty_factor Optional vector of penalty factors
#'
#' @return A list with best CV error, lambda, and alpha
#'
#' @keywords internal
cv_alpha <- function(X_std, y, standardize_X, alpha_sequence,
                     nlambda, lambda_min_ratio, num_folds, num_cores = 1,
                     penalty_factor = NULL) {

  if(is.null(penalty_factor)) penalty_factor <- rep(1, ncol(X_std))
  # Set up parallel backend
  doParallel::registerDoParallel(cores = num_cores)

  # Container for results
  results <- foreach::foreach(current_alpha = alpha_sequence, .combine = 'rbind', .multicombine = TRUE) %dopar% {
    cv_model <- glmnet::cv.glmnet(
      x = X_std, y = y,
      alpha = current_alpha,
      standardize = standardize_X,
      penalty.factor = penalty_factor,
      nfolds = num_folds,
      nlambda = nlambda,
      lambda.min.ratio = lambda_min_ratio
    )

    c(min(cv_model$cvm), cv_model$lambda.min, current_alpha)
  }

  # Find best alpha based on CV error
  best_idx <- which.min(results[, 1])

  return(list(
    best_cv_error = results[best_idx, 1],
    best_lambda = results[best_idx, 2],
    best_alpha = results[best_idx, 3]
  ))
}

#' Cross-validation to find optimal gamma parameter
#'
#' @param X_std Standardized design matrix
#' @param y Response vector
#' @param standardize_X Whether to standardize the predictors
#' @param alpha_sequence Sequence of alpha values to try
#' @param nlambda Number of lambda values
#' @param lambda_min_ratio Ratio of min to max lambda
#' @param gamma_sequence Sequence of gamma values to try
#' @param initial_weights Initial weights for adaptive penalties
#' @param num_folds Number of cross-validation folds
#' @param num_cores Number of cores for parallel computation
#'
#' @return A list with best CV error, lambda, alpha, and gamma
#'
#' @keywords internal
cv_gamma <- function(X_std, y, standardize_X, alpha_sequence, nlambda,
                     lambda_min_ratio, gamma_sequence,
                     initial_weights, num_folds, num_cores) {

  # Set up parallel backend
  doParallel::registerDoParallel(cores = num_cores)

  # Container for results
  results <- list()

  # For each gamma value
  for (i in seq_along(gamma_sequence)) {
    gamma <- gamma_sequence[i]
    adaptive_weights <- pmin(initial_weights^gamma, 1e+100)

    # Cross-validation across alpha values for the current gamma
    cv_results <- cv_alpha(X_std, y, standardize_X, alpha_sequence,
                           nlambda, lambda_min_ratio, num_folds, num_cores,
                           adaptive_weights)

    results[[i]] <- list(
      cv_error = cv_results$best_cv_error,
      lambda = cv_results$best_lambda,
      alpha = cv_results$best_alpha,
      gamma = gamma
    )
  }

  # Find best gamma
  best_idx <- which.min(sapply(results, function(x) x$cv_error))

  return(list(
    best_cv_error = results[[best_idx]]$cv_error,
    best_lambda = results[[best_idx]]$lambda,
    best_alpha = results[[best_idx]]$alpha,
    best_gamma = results[[best_idx]]$gamma
  ))
}

#' Cross-validation for covariate-dependent structure
#'
#' @param X_std Standardized design matrix
#' @param y Response vector
#' @param standardize_X Whether to standardize the predictors
#' @param alpha_sequence Sequence of alpha values to try
#' @param nlambda Number of lambda values
#' @param lambda_min_ratio Ratio of min to max lambda
#' @param gamma_sequence Sequence of gamma values to try
#' @param current_coefficients Current coefficient estimates
#' @param covariate_std Standardized covariate matrix
#' @param nvars Number of predictor variables
#' @param num_folds Number of cross-validation folds
#' @param num_cores Number of cores for parallel computation
#'
#' @return A list with best CV error, lambda, alpha, gamma, and optimal tau parameters
#'
#' @keywords internal
cv_covariate <- function(X_std, y, standardize_X, alpha_sequence, nlambda,
                         lambda_min_ratio, gamma_sequence,
                         current_coefficients, covariate_std, nvars,
                         num_folds, num_cores) {

  # Set up parallel backend
  doParallel::registerDoParallel(cores = num_cores)

  # Container for results
  results <- list()

  # For each gamma value
  for (i in seq_along(gamma_sequence)) {
    gamma <- gamma_sequence[i]

    # Optimize tau parameters for the current gamma
    optimal_tau_params <- optimize_tau(gamma, current_coefficients, covariate_std, nvars)

    # Calculate adaptive weights based on optimal tau parameters
    adaptive_weights <- pmin(exp(optimal_tau_params[1] + covariate_std%*%optimal_tau_params[-1]), 1e+100)

    # Cross-validation across alpha values for the current gamma and tau parameters
    cv_results <- cv_alpha(X_std, y, standardize_X, alpha_sequence,
                           nlambda, lambda_min_ratio, num_folds, num_cores,
                           adaptive_weights)

    results[[i]] <- list(
      cv_error = cv_results$best_cv_error,
      lambda = cv_results$best_lambda,
      alpha = cv_results$best_alpha,
      gamma = gamma,
      optimal_tau_params = optimal_tau_params
    )
  }

  # Find best gamma
  best_idx <- which.min(sapply(results, function(x) x$cv_error))

  return(list(
    best_cv_error = results[[best_idx]]$cv_error,
    best_lambda = results[[best_idx]]$lambda,
    best_alpha = results[[best_idx]]$alpha,
    best_gamma = results[[best_idx]]$gamma,
    optimal_tau_params = results[[best_idx]]$optimal_tau_params
  ))
}

#' Optimize tau parameters for covariate-dependent structure
#'
#' @param gamma Current gamma value
#' @param current_coefficients Current coefficient estimates
#' @param covariate_std Standardized covariate matrix
#' @param max_iter Maximum number of iterations for optimization (default: 5000)
#'
#' @return Vector of optimal tau parameters
#'
#' @keywords internal
optimize_tau <- function(gamma, current_coefficients, covariate_std, max_iter = 5000) {
  # Input validation
  if (!is.numeric(gamma) || length(gamma) != 1) {
    stop("gamma must be a single numeric value")
  }

  if (!is.numeric(current_coefficients)) {
    stop("current_coefficients must be numeric")
  }

  # Handle multi-dimensional covariates
  # If covariate_std is a vector, convert to matrix with one column
  if (is.vector(covariate_std)) {
    covariate_std <- as.matrix(covariate_std, ncol = 1)
  } else if (!is.matrix(covariate_std)) {
    stop("covariate_std must be a numeric vector or matrix")
  }

  # Get dimensions
  n_obs <- length(current_coefficients)
  n_covariates <- ncol(covariate_std)

  # Initialize starting values with zeros
  start_values <- rep(0, n_covariates + 1)

  # Set bounds for optimization
  lower_bounds <- rep(-30, n_covariates + 1)
  upper_bounds <- rep(30, n_covariates + 1)

  if (gamma == 1) {
    optimization_result <- stats::nlminb(
      start = start_values,
      lower = lower_bounds,
      upper = upper_bounds,
      objective = function(x) {
        # x[1] is the intercept, x[2:length(x)] are covariate coefficients
        tau_values <- x[1]
        if (n_covariates > 0) {
          for (j in 1:n_covariates) {
            tau_values <- tau_values + x[j + 1] * covariate_std[, j]
          }
        }
        sum(exp(tau_values) * abs(current_coefficients)) - sum(tau_values)
      },
      control = list(eval.max = max_iter, iter.max = max_iter)
    )
  } else if ((gamma > 0) && (gamma < 1)) {
    optimization_result <- stats::nlminb(
      start = start_values,
      lower = lower_bounds,
      upper = upper_bounds,
      objective = function(x) {
        # x[1] is the intercept, x[2:length(x)] are covariate coefficients
        tau_values <- x[1]
        if (n_covariates > 0) {
          for (j in 1:n_covariates) {
            tau_values <- tau_values + x[j + 1] * covariate_std[, j]
          }
        }
        sum(exp(tau_values) * abs(current_coefficients)) -
          (sum(exp((1 - (1/gamma)) * tau_values))) / (1 - (1/gamma))
      },
      control = list(eval.max = max_iter, iter.max = max_iter)
    )
  } else {
    stop("gamma must be a positive number between 0 and 1, or exactly 1")
  }

  # Return optimization results
  return(optimization_result$par)
}

