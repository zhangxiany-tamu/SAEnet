# This file helps manage imports from other packages and declare global variables
# to address R CMD check NOTES.

#-------------------------------------------------------------------------------
# Imports for packages explicitly used in your saenet function and helpers
#-------------------------------------------------------------------------------

#' @importFrom glmnet glmnet
#' @importFrom glmnet cv.glmnet
NULL

#' @importFrom foreach foreach
#' @importFrom foreach %dopar%
NULL

#' @importFrom doParallel registerDoParallel
NULL

#-------------------------------------------------------------------------------
# Imports from 'stats' package (base R)
#-------------------------------------------------------------------------------
# Functions from 'stats' used in your saenet.R code
#' @importFrom stats model.frame
#' @importFrom stats model.response
#' @importFrom stats model.matrix
#' @importFrom stats sd
#' @importFrom stats nlminb
#' @importFrom stats rnorm
#' @importFrom stats runif
NULL

#-------------------------------------------------------------------------------
# Imports from 'graphics' package (base R)
#-------------------------------------------------------------------------------
# Functions from 'graphics' used in your plot.saenet S3 method
#' @importFrom graphics abline
#' @importFrom graphics axis
#' @importFrom graphics barplot
#' @importFrom graphics dotchart
#' @importFrom graphics grid
#' @importFrom graphics layout
#' @importFrom graphics par
#' @importFrom graphics plot
#' @importFrom graphics points
#' @importFrom graphics text
# Ensure all graphics functions used in plot.saenet are listed here.
# The list above covers: par, dotchart, abline, layout, barplot, grid, plot, axis, points, text.
NULL

#-------------------------------------------------------------------------------
# Addressing NOTE for undefined global variables
#-------------------------------------------------------------------------------
# This declares 'current_alpha' as a global variable to R CMD check.
# This is often necessary for variables used in non-standard evaluation,
# such as in dplyr pipes, data.table, or with packages like 'foreach'.
# The variable 'current_alpha' is used as the iteration variable in the
# `foreach` loop within your `cv_alpha` function.
if (getRversion() >= "2.15.1") {
  utils::globalVariables(c(
    "current_alpha_iter"
    # Add any other undefined global variables identified by R CMD check here.
    # e.g. "alpha" if it was still an issue from previous checks.
  ))
}

