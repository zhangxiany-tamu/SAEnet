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
  cat("CV error:", x$cv_errors[length(x$cv_errors)], "\n")

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
#' Create plots for a fitted SAEnet model
#'
#' @param x A fitted SAEnet model object
#' @param type Character. Type of plot: "coefficients" for coefficient plot or "cv.error" for cross-validation error plot
#' @param iteration Integer. The iteration number to use for plotting coefficients (default: the final iteration)
#' @param top_n Integer. Maximum number of variables to display (ordered by absolute value). Default is NULL (all)
#' @param max_name_length Integer. Maximum length for variable names in plot. Longer names will be truncated.
#'                       Default is 20. Set to NULL to show full names.
#' @param ... Further arguments passed to the plotting function
#'
#' @return None (creates a plot)
#'
#' @export
plot.saenet <- function(x, type = c("coefficients", "cv.error"), iteration = NULL,
                        top_n = NULL, max_name_length = 20, ...) {
  type <- match.arg(type)

  if (is.null(iteration)) {
    iteration <- x$num_iterations
  }

  if (type == "coefficients") {
    # Get coefficient values from specified iteration
    beta <- as.numeric(x$coefficients[, iteration + 1])

    # Get variable names
    if (!is.null(rownames(x$coefficients))) {
      var_names <- rownames(x$coefficients)
    } else {
      var_names <- paste0("X", seq_along(beta))
    }

    names(beta) <- var_names

    # Only plot non-zero coefficients
    nonzero_idx <- which(abs(beta) > 1e-8)

    if (length(nonzero_idx) == 0) {
      message("No non-zero coefficients to plot")
      return(invisible())
    }

    beta_nonzero <- beta[nonzero_idx]
    var_names_nonzero <- var_names[nonzero_idx]

    # If top_n is specified, select top_n variables by absolute coefficient value
    if (!is.null(top_n) && top_n < length(beta_nonzero)) {
      top_idx <- order(abs(beta_nonzero), decreasing = TRUE)[1:top_n]
      beta_nonzero <- beta_nonzero[top_idx]
      var_names_nonzero <- var_names_nonzero[top_idx]
    }

    # Sort by coefficient value for visualization
    sorted_idx <- order(beta_nonzero)
    beta_sorted <- beta_nonzero[sorted_idx]
    var_names_sorted <- var_names_nonzero[sorted_idx]

    # Truncate variable names if they're too long
    if (!is.null(max_name_length) && max_name_length > 0) {
      long_names <- nchar(var_names_sorted) > max_name_length
      if (any(long_names)) {
        var_names_sorted[long_names] <- paste0(
          substr(var_names_sorted[long_names], 1, max_name_length - 3), "..."
        )
      }
    }

    # Calculate dimensions based on number of variables
    num_vars <- length(beta_sorted)
    plot_height <- max(7, min(3 + 0.25 * num_vars, 15))  # Dynamic height based on number of variables

    # Prepare alternative plot for large number of variables
    if (num_vars > 50) {
      # Use dotchart for large number of variables
      # Save current graphical parameters
      old_par <- par(no.readonly = TRUE)

      # Set up plot parameters
      par(mar = c(4, max(6, max(nchar(var_names_sorted)) * 0.4), 4, 2))

      # Create a dotchart - Fix: changed 'col' to 'color' for proper argument name
      dotchart(beta_sorted,
               labels = var_names_sorted,
               main = paste("Non-zero coefficients (iteration", iteration, ")"),
               xlab = "Coefficient value",
               pch = 19,
               color = ifelse(beta_sorted > 0, "steelblue", "firebrick"),
               cex = 0.8,
               pt.cex = 1.2)

      # Add a vertical line at x=0
      abline(v = 0, lty = 2, col = "gray50")

      # Restore original graphical parameters
      par(old_par)

    } else {
      # For smaller number of variables, use horizontal barplot
      # Use layout to create a plot with appropriate height
      layout_matrix <- matrix(1)
      layout(layout_matrix, heights = plot_height)

      # Set up margins for the horizontal barplot
      par(mar = c(4, max(6, max(nchar(var_names_sorted)) * 0.4), 4, 2))

      # Set up colors based on sign of coefficients
      colors <- ifelse(beta_sorted > 0, "steelblue", "firebrick")

      # Create barplot with variable names as labels
      bp <- barplot(beta_sorted,
                    main = paste("Non-zero coefficients (iteration", iteration, ")"),
                    col = colors,
                    horiz = TRUE,
                    las = 1,
                    names.arg = var_names_sorted,
                    xlab = "Coefficient value",
                    cex.names = 0.8,
                    xlim = c(min(beta_sorted) * 1.2, max(beta_sorted) * 1.2))

      # Add grid lines
      grid(nx = NULL, ny = 0, lty = 2, col = "gray80")

      # Add a vertical line at x=0
      abline(v = 0, lty = 2, col = "gray50")

      # Reset layout
      layout(1)
    }

  } else if (type == "cv.error") {
    # Plot CV error across iterations
    iterations <- 0:x$num_iterations
    cv_errors <- x$cv_errors

    # Set up margins for the CV error plot
    old_par <- par(no.readonly = TRUE)
    par(mar = c(4, 4, 4, 2))

    # Create the plot
    plot(iterations, cv_errors,
         type = "b",
         pch = 16,
         xlab = "Iteration",
         ylab = "Cross-validation error",
         main = "CV error versus iteration number",
         col = "steelblue",
         lwd = 2,
         xaxt = "n")  # Suppress default x-axis labels

    # Add custom x-axis with proper labels
    axis(1, at = iterations)

    # Add grid
    grid(nx = NULL, ny = NULL, lty = 2, col = "gray80")

    # Highlight best iteration (lowest CV error)
    best_iter <- which.min(cv_errors) - 1
    points(best_iter, cv_errors[best_iter + 1],
           col = "firebrick", cex = 2, pch = 16)
    text(best_iter, cv_errors[best_iter + 1],
         labels = paste("Best:", best_iter),
         pos = 3, col = "firebrick")

    # Restore original graphical parameters
    par(old_par)
  }

  # Return invisibly
  invisible(x)
}

