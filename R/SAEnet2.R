#' Structure-Adaptive Elastic Net using gcdnet
#'
#' @description
#' Implements the Structure-Adaptive Elastic Net (SAEnet) method using gcdnet, which extends
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
#' @param lambda2_sequence A sequence of L2 penalty parameters to try
#'        (default: exp(seq(0, -40, length.out = 50)))
#' @param nlambda Number of lambda values to generate in the sequence (default: 100)
#' @param lambda_selection_rule Character string specifying the rule for selecting the optimal lambda from
#'        cross-validation. Options are:
#'        \itemize{
#'          \item `"lambda.min"`: (Default) The lambda that gives the minimum mean cross-validated error.
#'          \item `"lambda.1se"`: The largest lambda such that the cross-validated error is within one
#'                standard error of the minimum. This typically results in a more parsimonious model.
#'          \item `"bic"`: Bayesian Information Criterion for model selection.
#'        }
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
#'           with columns corresponding to iterations
#'     \item weights: Matrix of adaptive weights for each iteration
#'     \item criterion_value: Vector of criterion values (CV error or BIC) for each iteration
#'           (corresponding to the selected lambda based on `lambda_selection_rule`)
#'     \item lambda: Vector of selected lambda parameters for each iteration
#'     \item lambda2: Vector of selected lambda2 parameters for each iteration
#'     \item gamma: Vector of selected gamma parameters for each iteration
#'     \item num_iterations: Number of iterations performed
#'     \item feature_scaling: Scaling information for predictors (if standardize=TRUE)
#'     \item lambda_selection_rule: The rule used for lambda selection.
#'   }
#'
#' @references
#' Pramanik, S., & Zhang, X. (2020). Structure Adaptive Elastic-Net. arXiv:2006.02041.
#'
#' @examples
#' # Generate common example data
#' set.seed(123)
#' n <- 100  # number of observations
#' p <- 50   # number of variables
#' X_data <- matrix(rnorm(n * p), n, p)
#' colnames(X_data) <- paste0("V", 1:p)
#' # True beta: first 5 are strong, next 5 are moderate, rest are zero
#' true_beta <- c(rep(1.5, 5), rep(0.3, 5), rep(0, p - 10))
#' y_data <- X_data %*% true_beta + rnorm(n, 0, 0.75) # Slightly more noise
#'
#' # --- Demonstrating the benefit of structure (using BIC for selection) ---
#' message("--- Comparing Models (BIC selection, max_iter=3, num_folds=3 for CV parts if any) ---")
#'
#' # 1. Standard SAEnet (No Structure)
#' fit_no_structure_bic <- saenet2(y = y_data, x = X_data, max_iterations = 3,
#'                                lambda_selection_rule = "bic",
#'                                num_cores = 1, verbose = FALSE, num_folds = 3)
#'
#' # 2. Group-based Structure
#' # Define groups: first 10 variables (true signals) in one group, rest in another
#' groups_meaningful <- c(rep(1, 10), rep(2, p - 10))
#' structure_group_meaningful <- list(group = groups_meaningful)
#' fit_group_bic <- saenet2(y = y_data, x = X_data,
#'                         structure_info = structure_group_meaningful,
#'                         max_iterations = 3, lambda_selection_rule = "bic",
#'                         num_cores = 1, verbose = FALSE, num_folds = 3)
#'
#' # 3. Covariate-dependent Structure
#' # Define a covariate: higher values for the first 10 true signal variables
#' covariate_meaningful <- rep(0.1, p)
#' covariate_meaningful[1:10] <- 1.0 # Higher value for true signals
#' covariate_meaningful[6:10] <- 0.7 # Moderately high for moderate signals
#' structure_cov_meaningful <- list(covariate = matrix(covariate_meaningful, ncol = 1))
#' fit_covariate_bic <- saenet2(y = y_data, x = X_data,
#'                             structure_info = structure_cov_meaningful,
#'                             max_iterations = 3, lambda_selection_rule = "bic",
#'                             num_cores = 1, verbose = FALSE, num_folds = 3)
#'
#' @export
saenet2 <- function(formula, data, x, y, structure_info, max_iterations = 5,
                          standardize = TRUE, lambda2_sequence = exp(seq(0, -40, length.out = 50)),
                          nlambda = 100, lambda_selection_rule = c("lambda.min", "lambda.1se", "bic"),
                          gamma_sequence = seq(0.1, 1, by = 0.1),
                          num_folds = 10, num_cores = 1, verbose = TRUE) {

  lambda_selection_rule <- match.arg(lambda_selection_rule)

  if (!requireNamespace("gcdnet", quietly = TRUE)) stop("Package gcdnet is required.")
  if (!requireNamespace("foreach", quietly = TRUE)) stop("Package foreach is required.")
  if (!requireNamespace("doParallel", quietly = TRUE)) stop("Package doParallel is required.")
  if (num_cores > 1 && !requireNamespace("parallel", quietly = TRUE)) stop("Package parallel is required for num_cores > 1.")

  if (!missing(formula)) {
    if (missing(data)) data <- environment(formula)
    model_frame <- stats::model.frame(formula, data)
    y_response <- stats::model.response(model_frame, "numeric")
    X_matrix <- stats::model.matrix(formula, model_frame)[, -1, drop = FALSE]
  } else if (!missing(x) && !missing(y)) {
    X_matrix <- as.matrix(x)
    y_response <- as.numeric(y)
  } else {
    stop("Either 'formula' and 'data' or 'x' and 'y' must be provided")
  }

  n_rows_X <- nrow(X_matrix)
  n_length_y <- length(y_response)
  if (n_rows_X != n_length_y) stop("Number of observations from y and X do not match.")
  nobs <- n_length_y

  if (missing(num_cores) || is.null(num_cores) || num_cores < 1) num_cores <- 1
  nvars <- ncol(X_matrix)
  if(nvars == 0) stop("The design matrix X has 0 columns (predictors).")

  if (standardize) {
    feature_means <- colMeans(X_matrix, na.rm = TRUE)
    feature_sds <- apply(X_matrix, 2, stats::sd, na.rm = TRUE)
    if(any(feature_sds == 0, na.rm = TRUE)) {
      if(verbose) message("Some predictors have zero standard deviation. They will not be scaled.")
      feature_sds[feature_sds == 0] <- 1
    }
    X_standardized <- scale(X_matrix, center = feature_means, scale = feature_sds)
  } else {
    X_standardized <- X_matrix
    feature_means <- rep(0, nvars)
    feature_sds <- rep(1, nvars)
  }

  fit_results <- fit_adaptive_net_gcdnet(
    y = y_response,
    X_std = X_standardized,
    structure_info = if(missing(structure_info)) NULL else structure_info,
    max_iterations = max_iterations,
    standardize_X_gcdnet = FALSE, # X_std is already handled
    lambda2_sequence = lambda2_sequence,
    nlambda = nlambda,
    lambda_selection_rule = lambda_selection_rule,
    gamma_sequence = gamma_sequence,
    num_folds = num_folds,
    num_cores = num_cores,
    verbose = verbose,
    nvars = nvars,
    nobs = nobs
  )

  coef_std_matrix <- fit_results$coef_std_matrix
  intercept_std_vector <- fit_results$intercept_std_vector
  weights_matrix <- fit_results$weights_matrix
  criterion_value_vector <- fit_results$criterion_value_vector
  lambda_vector <- fit_results$lambda_vector
  lambda2_vector <- fit_results$lambda2_vector
  gamma_vector <- fit_results$gamma_vector

  coef_matrix <- matrix(NA, nrow = nvars, ncol = ncol(coef_std_matrix))
  intercept_final_vector <- intercept_std_vector

  if (standardize) {
    for (j in 1:ncol(coef_std_matrix)) {
      coef_matrix[, j] <- coef_std_matrix[, j] / feature_sds
      intercept_final_vector[j] <- intercept_std_vector[j] - sum(coef_std_matrix[, j] * feature_means / feature_sds)
    }
  } else {
    coef_matrix <- coef_std_matrix
    intercept_final_vector <- intercept_std_vector
  }

  if (!is.null(colnames(X_matrix))) {
    rownames(coef_matrix) <- colnames(X_matrix)
  }

  result <- list(
    'intercept' = intercept_final_vector,
    'coefficients' = as.matrix(coef_matrix),
    'weights' = weights_matrix,
    'criterion_value' = criterion_value_vector,
    'lambda' = lambda_vector,
    'lambda2' = lambda2_vector,
    'gamma' = gamma_vector,
    'num_iterations' = max_iterations,
    'lambda_selection_rule' = lambda_selection_rule
  )

  if (standardize) {
    result$feature_scaling <- list(center = feature_means, scale = feature_sds)
  }
  class(result) <- c("saenet", "list")
  return(result)
}

#' Fit Structure-Adaptive Elastic Net model using gcdnet (Internal)
#'
#' @param y Response vector
#' @param X_std Standardized design matrix
#' @param structure_info External structural information
#' @param max_iterations Maximum number of iterations
#' @param standardize_X_gcdnet Logical, passed to gcdnet's standardize argument.
#'        Note: X_std is already standardized if saenet's standardize=TRUE.
#'        This controls gcdnet's internal standardization. Usually FALSE if X_std is pre-standardized.
#' @param lambda2_sequence Sequence of lambda2 values
#' @param nlambda Number of lambda values
#' @param lambda_selection_rule Rule for lambda selection ("lambda.min", "lambda.1se", or "bic")
#' @param gamma_sequence Sequence of gamma values
#' @param num_folds Number of cross-validation folds
#' @param num_cores Number of cores for parallel computation
#' @param verbose Whether to print progress messages
#' @param nvars Number of predictor variables
#' @param nobs Number of observations
#'
#' @return A list with model fitting results on the standardized scale.
#' @keywords internal
fit_adaptive_net_gcdnet <- function(y, X_std, structure_info, max_iterations, standardize_X_gcdnet,
                                    lambda2_sequence, nlambda, lambda_selection_rule,
                                    gamma_sequence, num_folds, num_cores, verbose, nvars, nobs) {

  if (is.null(structure_info)) {
    structure_type <- "standard"
  } else if (!is.null(names(structure_info)) && names(structure_info)[1] == 'group') {
    structure_type <- "group"
    if(is.null(structure_info$group) || length(structure_info$group) != nvars) stop("Group structure: structure_info$group length mismatch.")
    group_ids <- unique(structure_info$group)
    num_groups <- length(group_ids)
    group_indices <- vector(mode = 'list', length = num_groups)
    for (idx in 1:num_groups) group_indices[[idx]] <- which(structure_info$group == group_ids[idx])
  } else if (!is.null(names(structure_info)) && names(structure_info)[1] == 'covariate') {
    structure_type <- "covariate"
    covariate <- structure_info$covariate
    if (is.null(covariate)) stop("Covariate structure: structure_info$covariate missing.")
    if (!is.matrix(covariate) && !is.data.frame(covariate)) covariate <- as.matrix(covariate, ncol = 1)
    if(nrow(covariate) != nvars) stop("Covariate structure: structure_info$covariate row mismatch.")
    covariate_means <- colMeans(covariate, na.rm = TRUE)
    covariate_sds <- apply(covariate, 2, stats::sd, na.rm = TRUE)
    covariate_sds[covariate_sds == 0] <- 1
    covariate_std <- scale(covariate, center = covariate_means, scale = covariate_sds)
  } else {
    stop("Invalid structure_info.")
  }

  coef_std_matrix <- matrix(0, nrow = nvars, ncol = 1 + max_iterations)
  weights_matrix <- matrix(1, nrow = nvars, ncol = 1 + max_iterations)
  intercept_std_vector <- criterion_value_vector <- lambda_vector <- lambda2_vector <- gamma_vector <- rep(NA, 1 + max_iterations)

  if (num_cores > 1) {
    cl <- parallel::makeCluster(num_cores)
    doParallel::registerDoParallel(cl)
    on.exit(parallel::stopCluster(cl), add = TRUE)
  } else {
    foreach::registerDoSEQ()
  }

  for (iteration in 0:max_iterations) {
    if (verbose && iteration > 0) print(paste('SAEnet Iteration', iteration, 'of', max_iterations))
    current_iter_penalty_factor <- if (iteration == 0) rep(1, nvars) else weights_matrix[, iteration]

    if (iteration == 0) {
      results <- cv_lambda2(X_std, y, standardize_X_gcdnet = standardize_X_gcdnet,
                            lambda2_sequence, nlambda,
                            num_folds, num_cores, nobs,
                            penalty_factor = current_iter_penalty_factor,
                            lambda_selection_rule = lambda_selection_rule)

      best_lambda2 <- results$best_lambda2
      best_lambda <- results$best_lambda
      best_criterion_value <- results$best_criterion_value

      if(is.na(best_lambda) || is.na(best_lambda2)) stop("Iteration 0: CV/BIC failed.")

      model_fit <- gcdnet::gcdnet(x = X_std, y = y, method = "ls",
                                  standardize = standardize_X_gcdnet,
                                  lambda = best_lambda,
                                  lambda2 = best_lambda2)
      intercept_std_vector[iteration + 1] <- as.numeric(model_fit$b0)
      coef_std_matrix[, iteration + 1] <- as.numeric(as.matrix(model_fit$beta))
      criterion_value_vector[iteration + 1] <- best_criterion_value
      lambda_vector[iteration + 1] <- best_lambda
      lambda2_vector[iteration + 1] <- best_lambda2
      weights_matrix[, iteration + 1] <- current_iter_penalty_factor
    } else {
      prev_coefs_std <- coef_std_matrix[, iteration]
      if (structure_type == "standard") {
        initial_weights_for_gamma_opt <- calculate_standard_weights(prev_coefs_std)
      } else if (structure_type == "group") {
        initial_weights_for_gamma_opt <- calculate_group_weights(prev_coefs_std, group_indices, num_groups)
      } else if (structure_type == "covariate") {
        cv_results_cov <- cv_covariate_gcdnet(X_std, y, standardize_X_gcdnet = standardize_X_gcdnet,
                                              lambda2_sequence, nlambda,
                                              lambda_selection_rule, gamma_sequence, prev_coefs_std,
                                              covariate_std, nvars, num_folds, num_cores, nobs, verbose)
        final_penalty_factor_for_iter <- cv_results_cov$optimal_weights_for_gcdnet
        model_fit <- gcdnet::gcdnet(x = X_std, y = y, method = "ls",
                                    standardize = standardize_X_gcdnet,
                                    pf = final_penalty_factor_for_iter,
                                    lambda = cv_results_cov$best_lambda,
                                    lambda2 = cv_results_cov$best_lambda2)
        intercept_std_vector[iteration + 1] <- as.numeric(model_fit$b0)
        coef_std_matrix[, iteration + 1] <- as.numeric(as.matrix(model_fit$beta))
        weights_matrix[, iteration + 1] <- final_penalty_factor_for_iter
        criterion_value_vector[iteration + 1] <- cv_results_cov$best_criterion_value
        lambda_vector[iteration + 1] <- cv_results_cov$best_lambda
        lambda2_vector[iteration + 1] <- cv_results_cov$best_lambda2
        gamma_vector[iteration + 1] <- cv_results_cov$best_gamma
        next
      }

      if (all(prev_coefs_std == 0) || any(!is.finite(initial_weights_for_gamma_opt))) {
        if(verbose) message(paste("Iter", iteration, ": Prev coefs zero or non-finite weights. Coefs set to zero."))
        coef_std_matrix[, iteration + 1] <- numeric(nvars)
        intercept_std_vector[iteration + 1] <- intercept_std_vector[iteration]
        weights_matrix[, iteration + 1] <- if(all(prev_coefs_std == 0)) rep(1e6, nvars) else initial_weights_for_gamma_opt
        criterion_value_vector[iteration + 1] <- criterion_value_vector[iteration]
        lambda_vector[iteration + 1] <- lambda_vector[iteration]
        lambda2_vector[iteration + 1] <- lambda2_vector[iteration]
        gamma_vector[iteration + 1] <- NA
        next
      }

      cv_results_gamma <- cv_gamma_gcdnet(X_std, y, standardize_X_gcdnet = standardize_X_gcdnet,
                                          lambda2_sequence, nlambda,
                                          lambda_selection_rule, gamma_sequence, initial_weights_for_gamma_opt,
                                          num_folds, num_cores, nobs, verbose)
      final_penalty_factor_for_iter <- pmin(initial_weights_for_gamma_opt^cv_results_gamma$best_gamma, 1e+12)
      final_penalty_factor_for_iter[!is.finite(final_penalty_factor_for_iter)] <- 1e+12
      model_fit <- gcdnet::gcdnet(x = X_std, y = y, method = "ls",
                                  standardize = standardize_X_gcdnet,
                                  pf = final_penalty_factor_for_iter,
                                  lambda = cv_results_gamma$best_lambda,
                                  lambda2 = cv_results_gamma$best_lambda2)
      intercept_std_vector[iteration + 1] <- as.numeric(model_fit$b0)
      coef_std_matrix[, iteration + 1] <- as.numeric(as.matrix(model_fit$beta))
      weights_matrix[, iteration + 1] <- final_penalty_factor_for_iter
      criterion_value_vector[iteration + 1] <- cv_results_gamma$best_criterion_value
      lambda_vector[iteration + 1] <- cv_results_gamma$best_lambda
      lambda2_vector[iteration + 1] <- cv_results_gamma$best_lambda2
      gamma_vector[iteration + 1] <- cv_results_gamma$best_gamma
    }
  }
  return(list(coef_std_matrix = coef_std_matrix, intercept_std_vector = intercept_std_vector,
              weights_matrix = weights_matrix, criterion_value_vector = criterion_value_vector,
              lambda_vector = lambda_vector, lambda2_vector = lambda2_vector, gamma_vector = gamma_vector))
}

#' Cross-validation across lambda2 values (Internal)
#'
#' @param X_std Standardized design matrix
#' @param y Response vector
#' @param standardize_X_gcdnet Logical, passed to gcdnet's standardize argument.
#' @param lambda2_sequence Sequence of lambda2 values
#' @param nlambda Number of lambda values
#' @param num_folds Number of CV folds
#' @param num_cores Number of cores
#' @param nobs Number of observations
#' @param penalty_factor Penalty factors for gcdnet
#' @param lambda_selection_rule Rule for lambda selection
#'
#' @return List with best criterion value, best lambda, best lambda2.
#' @keywords internal
cv_lambda2 <- function(X_std, y, standardize_X_gcdnet, lambda2_sequence,
                       nlambda, num_folds, num_cores, nobs,
                       penalty_factor = NULL, lambda_selection_rule) {

  if(is.null(penalty_factor)) penalty_factor <- rep(1, ncol(X_std))
  penalty_factor[!is.finite(penalty_factor)] <- 1e+12

  results_list <- foreach::foreach(
    current_lambda2_iter = lambda2_sequence,
    .combine = 'rbind', .errorhandling = "pass", .packages = "gcdnet"
  ) %dopar% {
    criterion_val <- NA
    selected_lambda_val <- NA

    if (lambda_selection_rule %in% c("lambda.min", "lambda.1se")) {
      cv_model_fit <- NULL
      try_result <- try({
        cv_model_fit <- gcdnet::cv.gcdnet(x = X_std, y = y, method = "ls",
                                          standardize = standardize_X_gcdnet,
                                          pf = penalty_factor, nfolds = num_folds,
                                          lambda2 = current_lambda2_iter,
                                          nlambda = nlambda)
      }, silent = TRUE)

      if (inherits(try_result, "try-error") || is.null(cv_model_fit)) {
        return(c(Inf, NA, current_lambda2_iter)) # criterion_val, lambda, lambda2
      }

      if (lambda_selection_rule == "lambda.min") {
        selected_lambda_val <- cv_model_fit$lambda.min
        idx_selected <- which(cv_model_fit$lambda == selected_lambda_val)
        if(length(idx_selected) > 0) criterion_val <- cv_model_fit$cvm[idx_selected[1]]
      } else { # lambda.1se
        selected_lambda_val <- cv_model_fit$lambda.1se
        idx_selected <- which(cv_model_fit$lambda == selected_lambda_val)
        if(length(idx_selected) > 0) criterion_val <- cv_model_fit$cvm[idx_selected[1]]
      }
      if(is.na(selected_lambda_val) || is.na(criterion_val)){ # Fallback if 1se is weird or index fails
        selected_lambda_val <- cv_model_fit$lambda.min
        idx_fallback <- which(cv_model_fit$lambda == selected_lambda_val)
        criterion_val <- if(length(idx_fallback)>0) cv_model_fit$cvm[idx_fallback[1]] else min(cv_model_fit$cvm, na.rm=TRUE)
      }

    } else if (lambda_selection_rule == "bic") {
      glm_fit <- NULL
      try_result_bic <- try({
        glm_fit <- gcdnet::gcdnet(x = X_std, y = y, method = "ls",
                                  standardize = standardize_X_gcdnet,
                                  pf = penalty_factor,
                                  lambda2 = current_lambda2_iter,
                                  nlambda = nlambda)
      }, silent = TRUE)

      if (inherits(try_result_bic, "try-error") || is.null(glm_fit) || length(glm_fit$lambda) == 0) {
        return(c(Inf, NA, current_lambda2_iter)) # criterion_val (BIC), lambda, lambda2
      }

      # Calculate BIC for each lambda in the path
      # BIC = deviance + log(nobs) * df
      # deviance = nulldev * (1 - dev.ratio)
      deviance_vals <- apply((X_std%*%glm_fit$beta - y)^2,2,mean)*nobs
      bic_vals <- deviance_vals + log(nobs) * glm_fit$df

      if(all(!is.finite(bic_vals))) { # If all BICs are Inf/NA
        return(c(Inf, NA, current_lambda2_iter))
      }

      min_bic_idx <- which.min(bic_vals)
      criterion_val <- bic_vals[min_bic_idx]
      selected_lambda_val <- glm_fit$lambda[min_bic_idx]
    } else {
      stop("Unknown lambda_selection_rule in cv_lambda2.")
    }
    c(criterion_val, selected_lambda_val, current_lambda2_iter)
  }

  results_df <- as.data.frame(results_list)
  colnames(results_df) <- c("criterion", "lambda", "lambda2")
  valid_results <- results_df[is.finite(results_df$criterion) & is.finite(results_df$lambda), ]

  if (nrow(valid_results) == 0) {
    stop(paste0("cv_lambda2: Selection rule '", lambda_selection_rule, "' failed for all lambda2 values."))
  }
  best_idx_in_valid <- which.min(valid_results$criterion) # Minimize CV error or BIC

  return(list(
    best_criterion_value = valid_results$criterion[best_idx_in_valid],
    best_lambda = valid_results$lambda[best_idx_in_valid],
    best_lambda2 = valid_results$lambda2[best_idx_in_valid]
  ))
}


#' Cross-validation to find optimal gamma parameter using gcdnet (Internal)
#' @keywords internal
cv_gamma_gcdnet <- function(X_std, y, standardize_X_gcdnet, lambda2_sequence, nlambda, lambda_selection_rule,
                            gamma_sequence, initial_weights_for_gamma_opt,
                            num_folds, num_cores, nobs, verbose) {
  gamma_cv_results_list <- list()
  for (i in seq_along(gamma_sequence)) {
    current_gamma_iter <- gamma_sequence[i]
    if(verbose > 1) print(paste("  cv_gamma_gcdnet: Testing gamma =", current_gamma_iter))
    current_penalty_factor_for_lambda2_cv <- pmin(initial_weights_for_gamma_opt^current_gamma_iter, 1e+12)
    current_penalty_factor_for_lambda2_cv[!is.finite(current_penalty_factor_for_lambda2_cv)] <- 1e+12

    lambda2_cv_results <- cv_lambda2(X_std, y, standardize_X_gcdnet, lambda2_sequence,
                                     nlambda, num_folds, num_cores, nobs,
                                     penalty_factor = current_penalty_factor_for_lambda2_cv,
                                     lambda_selection_rule = lambda_selection_rule)
    gamma_cv_results_list[[i]] <- list(
      criterion_value = lambda2_cv_results$best_criterion_value,
      lambda = lambda2_cv_results$best_lambda,
      lambda2 = lambda2_cv_results$best_lambda2,
      gamma = current_gamma_iter)
  }
  criterion_values_for_gamma <- sapply(gamma_cv_results_list, function(x) x$criterion_value)
  if(all(!is.finite(criterion_values_for_gamma))) stop("cv_gamma_gcdnet: All gammas non-finite criterion.")
  best_gamma_idx_val <- which.min(criterion_values_for_gamma)

  return(list(
    best_criterion_value = gamma_cv_results_list[[best_gamma_idx_val]]$criterion_value,
    best_lambda = gamma_cv_results_list[[best_gamma_idx_val]]$lambda,
    best_lambda2 = gamma_cv_results_list[[best_gamma_idx_val]]$lambda2,
    best_gamma = gamma_cv_results_list[[best_gamma_idx_val]]$gamma ))
}

#' Cross-validation for covariate-dependent structure using gcdnet (Internal)
#' @keywords internal
cv_covariate_gcdnet <- function(X_std, y, standardize_X_gcdnet, lambda2_sequence, nlambda,
                                lambda_selection_rule, gamma_sequence, prev_coefs_std,
                                covariate_std, nvars, num_folds, num_cores, nobs, verbose) {
  covariate_cv_results_list <- list()
  for (i in seq_along(gamma_sequence)) {
    current_gamma_for_tau_opt <- gamma_sequence[i]
    if(verbose > 1) print(paste("  cv_covariate_gcdnet: Testing gamma_tau =", current_gamma_for_tau_opt))
    optimal_tau_params_val <- optimize_tau(current_gamma_for_tau_opt, prev_coefs_std, covariate_std, nvars, verbose = verbose)
    linear_pred_tau <- optimal_tau_params_val[1]
    if(ncol(covariate_std) > 0) linear_pred_tau <- linear_pred_tau + covariate_std %*% optimal_tau_params_val[-1]
    current_penalty_factor_for_lambda2_cv <- pmin(exp(linear_pred_tau), 1e+12)
    current_penalty_factor_for_lambda2_cv[!is.finite(current_penalty_factor_for_lambda2_cv)] <- 1e+12

    lambda2_cv_results_cov <- cv_lambda2(X_std, y, standardize_X_gcdnet, lambda2_sequence,
                                         nlambda, num_folds, num_cores, nobs,
                                         penalty_factor = current_penalty_factor_for_lambda2_cv,
                                         lambda_selection_rule = lambda_selection_rule)
    covariate_cv_results_list[[i]] <- list(
      criterion_value = lambda2_cv_results_cov$best_criterion_value,
      lambda = lambda2_cv_results_cov$best_lambda,
      lambda2 = lambda2_cv_results_cov$best_lambda2,
      gamma_for_tau = current_gamma_for_tau_opt,
      optimal_tau_params = optimal_tau_params_val,
      final_weights = current_penalty_factor_for_lambda2_cv)
  }
  criterion_values_for_cov_gamma <- sapply(covariate_cv_results_list, function(x) x$criterion_value)
  if(all(!is.finite(criterion_values_for_cov_gamma))) stop("cv_covariate_gcdnet: All gamma_tau non-finite criterion.")
  best_cov_gamma_idx_val <- which.min(criterion_values_for_cov_gamma)

  return(list(
    best_criterion_value = covariate_cv_results_list[[best_cov_gamma_idx_val]]$criterion_value,
    best_lambda = covariate_cv_results_list[[best_cov_gamma_idx_val]]$lambda,
    best_lambda2 = covariate_cv_results_list[[best_cov_gamma_idx_val]]$lambda2,
    best_gamma = covariate_cv_results_list[[best_cov_gamma_idx_val]]$gamma_for_tau,
    optimal_tau_params = covariate_cv_results_list[[best_cov_gamma_idx_val]]$optimal_tau_params,
    optimal_weights_for_gcdnet = covariate_cv_results_list[[best_cov_gamma_idx_val]]$final_weights ))
}
