% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{predict.saenet}
\alias{predict.saenet}
\title{Predict from a Structure-Adaptive Elastic Net Model}
\usage{
\method{predict}{saenet}(
  object,
  newx,
  type = c("response", "coefficients"),
  iteration = NULL,
  ...
)
}
\arguments{
\item{object}{A fitted SAEnet model object}

\item{newx}{A matrix of new data for prediction (required for type="response")}

\item{type}{Character. Type of prediction: "response" for predictions or "coefficients" for model coefficients}

\item{iteration}{Integer. The iteration number to use for prediction (default: the final iteration)}

\item{...}{Further arguments (not used)}
}
\value{
A vector of predicted values (for type="response") or coefficient values (for type="coefficients")
}
\description{
Predict responses or extract coefficients from a fitted SAEnet model
}
