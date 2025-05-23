% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SAEnet.R
\name{cv_covariate}
\alias{cv_covariate}
\title{Cross-validation for covariate-dependent structure (Internal)}
\usage{
cv_covariate(
  X_std,
  y,
  standardize_X_glmnet,
  alpha_sequence,
  nlambda,
  lambda_min_ratio,
  lambda_selection_rule,
  gamma_sequence,
  prev_coefs_std,
  covariate_std,
  nvars,
  num_folds,
  num_cores,
  nobs,
  verbose
)
}
\description{
Cross-validation for covariate-dependent structure (Internal)
}
\keyword{internal}
