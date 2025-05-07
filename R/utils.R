#' Print a Structure-Adaptive Elastic Net Model
#'
#' @description
#' Prints a summary of a fitted Structure-Adaptive Elastic Net model.
#'
#' @param x A fitted SAEnet model object
#' @param ... Further arguments (not used)
#'
#' @return Invisibly returns the input object
#'
#' @export
print.saenet <- function(x, ...) {
  cat("Structure-Adaptive Elastic Net\n")
  cat("------------------------------\n")
  cat("Number of iterations:", x$num_iterations, "\n")
  cat("Number of non-zero coefficients:", sum(abs(x$coefficients[, ncol(x$coefficients)]) > 1e-8), "\n")
  cat("\nFinal model parameters:\n")
  cat("Lambda:", x$lambda[length(x$lambda)], "\n")
  cat("Alpha:", x$alpha[length(x$alpha)], "\n")
  if (!is.na(x$gamma[length(x$gamma)])) {
    cat("Gamma:", x$gamma[length(x$gamma)], "\n")
  }
  cat("CV error:", x$criterion_value[length(x$criterion_value)], "\n")

  # Return invisibly
  invisible(x)
}

#' Predict from a Structure-Adaptive Elastic Net Model
#'
#' @description
#' Predict responses or extract coefficients from a fitted SAEnet model
#'
#' @param object A fitted SAEnet model object
#' @param newx A matrix of new data for prediction (required for type="response")
#' @param type Character. Type of prediction: "response" for predictions or "coefficients" for model coefficients
#' @param iteration Integer. The iteration number to use for prediction (default: the final iteration)
#' @param ... Further arguments (not used)
#'
#' @return A vector of predicted values (for type="response") or coefficient values (for type="coefficients")
#'
#' @export
predict.saenet <- function(object, newx, type = c("response", "coefficients"), iteration = NULL, ...) {
  type <- match.arg(type)

  # If iteration is not specified, use the last one
  if (is.null(iteration)) {
    iteration <- object$num_iterations
  } else if (iteration < 0 || iteration > object$num_iterations) {
    stop("Invalid iteration number")
  }

  if (type == "coefficients") {
    return(object$coefficients[, iteration + 1])
  } else if (type == "response") {
    if (missing(newx)) {
      stop("newx is required for prediction")
    }

    newx <- as.matrix(newx)
    beta <- object$coefficients[, iteration + 1]
    intercept <- object$intercept[iteration + 1]

    return(intercept + newx %*% beta)
  }
}

#' Plot a Structure-Adaptive Elastic Net Model
#'
#' @description
#' Create plots for a fitted SAEnet model object. This function can plot
#' the coefficient paths or the cross-validation error across iterations.
#'
#' @param x A fitted SAEnet model object (output from the `saenet` function).
#' @param type Character string. Specifies the type of plot:
#'   \itemize{
#'     \item `"coefficients"`: (Default) Plots the non-zero coefficient values from a
#'           specified iteration. Uses a horizontal barplot for fewer than or equal to 50
#'           coefficients and a dotchart for more than 50.
#'     \item `"criterion.value"`: Plots the mean cross-validation error against the
#'           iteration number.
#'   }
#' @param iteration Integer. The iteration number from the SAEnet fit to use for
#'   plotting coefficients. Defaults to the final iteration (`x$num_iterations`).
#'   Not used if `type = "criterion.value"`.
#' @param top_n Integer or NULL. For `type = "coefficients"`, specifies the maximum
#'   number of non-zero variables to display, ordered by the absolute value of their
#'   coefficients. If NULL (default), all non-zero coefficients are plotted.
#' @param max_name_length Integer or NULL. For `type = "coefficients"`, the maximum
#'   length for variable names displayed in the plot. Longer names will be truncated
#'   and appended with "...". Default is 20. Set to NULL to show full names without truncation.
#' @param ... Further arguments passed to the underlying plotting functions
#'   (e.g., `graphics::barplot`, `graphics::dotchart`, `graphics::plot`).
#'
#' @return Invisibly returns the input object `x`. The function's primary purpose
#'   is to generate a plot.
#'
#' @method plot saenet
#' @export
#'
#' @examples
#' # Generate some example data (as in the saenet function examples)
#' set.seed(123)
#' n_obs <- 100
#' n_vars <- 20 # Using fewer variables for a quicker plot example
#' X_example <- matrix(rnorm(n_obs * n_vars), n_obs, n_vars)
#' colnames(X_example) <- paste0("Var", 1:n_vars)
#' true_beta_example <- c(rep(1.5, 5), rep(-0.8, 3), rep(0, n_vars - 8))
#' y_example <- X_example %*% true_beta_example + rnorm(n_obs, 0, 0.5)
#'
#' # Assuming 'saenet' function and its dependencies are loaded
#' # And assuming 'predict.saenet' is also defined for the examples in saenet()
#' # For this plot example, we just need a dummy saenet object structure
#' # if the full saenet() call is too slow for a simple plot test.
#'
#' # Dummy saenet fit for plotting examples (replace with actual fit if saenet is fast)
#' if (requireNamespace("stats", quietly = TRUE) && exists("saenet")) {
#'   # Use a very minimal saenet run if possible
#'   # This is just to ensure the examples can run if saenet() itself is not defined here
#'   # In a real package, saenet() would be available.
#'   capture.output( # Suppress verbose output from saenet for example
#'     fit_example <- tryCatch(
#'        saenet(y = y_example, x = X_example, max_iterations = 2,
#'               num_folds = 3, num_cores = 1, verbose = FALSE,
#'               lambda_selection_rule = "lambda.min"),
#'        error = function(e) NULL
#'     )
#'   )
#'
#'   if (!is.null(fit_example)) {
#'     # Plot non-zero coefficients from the final iteration
#'     plot(fit_example, type = "coefficients")
#'
#'     # Plot top 5 non-zero coefficients
#'     plot(fit_example, type = "coefficients", top_n = 5)
#'
#'     # Plot CV error across iterations
#'     plot(fit_example, type = "criterion.value")
#'   } else {
#'     message("Skipping plot.saenet examples as saenet() function or
#'     its dependencies are not fully available or failed.")
#'   }
#' } else {
#'   message("Skipping plot.saenet examples: 'stats' package or 'saenet' function not available.")
#' }
#'
plot.saenet <- function(x, type = c("coefficients", "criterion.value"), iteration = NULL,
                        top_n = NULL, max_name_length = 20, ...) {
  # Validate input type
  type <- match.arg(type)

  # Save current graphical parameters and restore on exit
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)

  # Determine the iteration to use for coefficient plots
  if (is.null(iteration)) {
    iteration <- x$num_iterations # This should be the actual number of iterations performed
  } else {
    if (!is.numeric(iteration) || iteration < 0 || iteration > x$num_iterations) {
      stop(paste0("Invalid iteration number. Must be between 0 and ", x$num_iterations, "."))
    }
  }

  if (type == "coefficients") {
    # Get coefficient values from the specified iteration (0-indexed in algorithm, 1-indexed in matrix)
    if ((iteration + 1) > ncol(x$coefficients)) {
      stop(paste0("Iteration ", iteration, " (column ", iteration + 1, ") not found in coefficients matrix."))
    }
    beta <- as.numeric(x$coefficients[, iteration + 1])

    # Get variable names
    if (!is.null(rownames(x$coefficients))) {
      var_names <- rownames(x$coefficients)
    } else {
      var_names <- paste0("X", seq_along(beta))
    }
    names(beta) <- var_names

    # Only consider non-zero coefficients
    nonzero_idx <- which(abs(beta) > 1e-8) # Tolerance for non-zero

    if (length(nonzero_idx) == 0) {
      message("No non-zero coefficients to plot for iteration ", iteration, ".")
      return(invisible(x))
    }

    beta_nonzero <- beta[nonzero_idx]
    var_names_nonzero <- var_names[nonzero_idx]

    # If top_n is specified, select top_n variables by absolute coefficient value
    if (!is.null(top_n) && is.numeric(top_n) && top_n > 0 && top_n < length(beta_nonzero)) {
      top_idx <- order(abs(beta_nonzero), decreasing = TRUE)[1:top_n]
      beta_nonzero <- beta_nonzero[top_idx]
      var_names_nonzero <- var_names_nonzero[top_idx]
    }

    # Sort by coefficient value for visualization (ascending)
    sorted_idx <- order(beta_nonzero)
    beta_sorted <- beta_nonzero[sorted_idx]
    var_names_sorted <- var_names_nonzero[sorted_idx]

    # Truncate variable names if they're too long
    if (!is.null(max_name_length) && max_name_length > 0) {
      long_names_idx <- nchar(var_names_sorted) > max_name_length
      if (any(long_names_idx)) {
        var_names_sorted[long_names_idx] <- paste0(
          substr(var_names_sorted[long_names_idx], 1, max_name_length - 3), "..."
        )
      }
    }

    num_vars_to_plot <- length(beta_sorted)
    plot_title <- paste("Non-zero Coefficients (Iteration", iteration, ")")

    # Determine x-axis limits for coefficient plots, ensuring 0 is included and there's padding
    x_axis_range <- range(0, beta_sorted, na.rm = TRUE)
    padding <- diff(x_axis_range) * 0.15
    if (padding == 0 && x_axis_range[1] == 0) { # Handles case where all betas are 0 (already caught) or single non-zero value
      padding <- max(abs(x_axis_range)) * 0.15 # If single value, pad relative to its magnitude
      if(padding == 0) padding <- 0.1 # Absolute fallback padding
    } else if (padding == 0) { # If range is zero but not at zero (e.g. all betas are 1)
      padding <- 0.1
    }
    final_xlim <- c(x_axis_range[1] - padding, x_axis_range[2] + padding)


    # Set common margin: left margin depends on max variable name length
    # mar = c(bottom, left, top, right)
    # Dynamically adjust left margin based on the length of variable names
    max_char_len <- if (num_vars_to_plot > 0) max(nchar(var_names_sorted), na.rm = TRUE) else 5
    left_margin_lines <- max(4, min(10, ceiling(max_char_len * 0.45) + 1)) # Adjusted factor and added base
    graphics::par(mar = c(5, left_margin_lines, 4, 2) + 0.1)


    if (num_vars_to_plot > 35) { # Threshold for switching to dotchart (was 50)
      # Use dotchart for a large number of variables
      point_colors <- ifelse(beta_sorted > 0, "steelblue", "firebrick")
      graphics::dotchart(beta_sorted,
                         labels = var_names_sorted,
                         main = plot_title,
                         xlab = "Coefficient Value",
                         pch = 19,
                         col = point_colors, # Use col for dotchart points
                         cex = 0.8, # cex for labels
                         pt.cex = 1.2, # pt.cex for point size
                         xlim = final_xlim,
                         ...)
      graphics::abline(v = 0, lty = 2, col = "gray50")
    } else {
      # For smaller number of variables, use horizontal barplot
      bar_colors <- ifelse(beta_sorted > 0, "steelblue", "firebrick")

      # Calculate bar positions for adding text if needed, though barplot handles names.arg
      bp_positions <- graphics::barplot(beta_sorted,
                                        main = plot_title,
                                        col = bar_colors,
                                        horiz = TRUE,
                                        las = 1, # Horizontal labels on y-axis
                                        names.arg = var_names_sorted,
                                        xlab = "Coefficient Value",
                                        cex.names = 0.8, # Size of variable names
                                        xlim = final_xlim,
                                        border = grDevices::adjustcolor("black", alpha.f = 0.5), # Add slight border to bars
                                        ...)
      # Add grid lines (vertical lines for horizontal barplot)
      graphics::grid(nx = NULL, ny = NA, lty = 2, col = grDevices::adjustcolor("gray", alpha.f = 0.5))
      graphics::abline(v = 0, lty = 2, col = "gray50") # Line at zero
      graphics::box() # Redraw box around plot
    }

  } else if (type == "criterion.value") {
    # Plot CV error across iterations
    # Iterations are 0-indexed in the algorithm, results stored 1-indexed
    iterations_to_plot <- 0:x$num_iterations

    if (length(x$criterion_value) != (x$num_iterations + 1)) {
      message("Length of criterion_value does not match num_iterations. Plotting available errors.")
      # Adjust iterations_to_plot if criterion_value is shorter (e.g. early stop not fully implemented)
      iterations_to_plot <- 0:(length(x$criterion_value)-1)
      if(length(iterations_to_plot) == 0) {
        message("No CV errors to plot.")
        return(invisible(x))
      }
    }
    criterion_value_to_plot <- x$criterion_value[1:length(iterations_to_plot)]


    graphics::par(mar = c(5, 4, 4, 2) + 0.1) # Default margins often fine here

    graphics::plot(iterations_to_plot, criterion_value_to_plot,
                   type = "b", # Both points and lines
                   pch = 16,
                   xlab = "Iteration Number",
                   ylab = "Mean Cross-Validation Error",
                   main = "CV Error vs. Iteration Number",
                   col = "steelblue",
                   lwd = 2,
                   xaxt = "n", # Suppress default x-axis
                   ...)

    # Add custom x-axis with integer labels for iterations
    graphics::axis(1, at = iterations_to_plot, labels = iterations_to_plot)
    graphics::grid(nx = NA, ny = NULL, lty = 2, col = grDevices::adjustcolor("gray", alpha.f = 0.5)) # Horizontal grid lines

    # Highlight the iteration with the minimum CV error
    if(any(is.finite(criterion_value_to_plot))){
      min_cv_error_idx <- which.min(criterion_value_to_plot)
      best_iteration_val <- iterations_to_plot[min_cv_error_idx]
      min_cv_error_val <- criterion_value_to_plot[min_cv_error_idx]

      graphics::points(best_iteration_val, min_cv_error_val,
                       col = "firebrick", cex = 2, pch = 16)
      graphics::text(best_iteration_val, min_cv_error_val,
                     labels = paste("Best Iteration:", best_iteration_val),
                     pos = 3, col = "firebrick", cex = 0.9) # pos=3 for above
    }
  }

  return(invisible(x))
}
