% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SAEnet.R
\name{cv_gamma}
\alias{cv_gamma}
\title{Cross-validation to find optimal gamma parameter (Internal)}
\usage{
cv_gamma(
  X_std,
  y,
  standardize_X_glmnet,
  alpha_sequence,
  nlambda,
  lambda_min_ratio,
  lambda_selection_rule,
  gamma_sequence,
  initial_weights_for_gamma_opt,
  num_folds,
  num_cores,
  nobs,
  verbose
)
}
\description{
Cross-validation to find optimal gamma parameter (Internal)
}
\keyword{internal}
