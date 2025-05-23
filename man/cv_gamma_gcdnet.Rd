% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SAEnet2.R
\name{cv_gamma_gcdnet}
\alias{cv_gamma_gcdnet}
\title{Cross-validation to find optimal gamma parameter using gcdnet (Internal)}
\usage{
cv_gamma_gcdnet(
  X_std,
  y,
  standardize_X_gcdnet,
  lambda2_sequence,
  nlambda,
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
Cross-validation to find optimal gamma parameter using gcdnet (Internal)
}
\keyword{internal}
