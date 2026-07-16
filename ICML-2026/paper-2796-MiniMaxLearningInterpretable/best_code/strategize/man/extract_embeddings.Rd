% Generated manually for embedding API
\name{extract_embeddings}
\alias{extract_embeddings}
\alias{extract_embeddings.strategic_predictor}
\alias{extract_embeddings.conjoint_foundation_model}
\title{Extract last-layer embeddings from fitted neural objects}
\usage{
extract_embeddings(object, ...)

\method{extract_embeddings}{strategic_predictor}(
  object,
  newdata,
  level = c("candidate", "respondent_context", "all"),
  factor_schema = NULL,
  text_embedding_fn = NULL,
  ...
)

\method{extract_embeddings}{conjoint_foundation_model}(
  object,
  newdata,
  group_key = NULL,
  mode = c("auto", "pairwise", "single"),
  likelihood = NULL,
  n_outcomes = NULL,
  level = c("candidate", "respondent_context", "all"),
  experiment_id = NULL,
  names_list = NULL,
  p_list = NULL,
  canonical_factor_id = NULL,
  canonical_level_id = NULL,
  foundation_adaptation_control = NULL,
  conda_env = "strategize_env",
  conda_env_required = TRUE,
  ...
)
}
\arguments{
\item{object}{A fitted neural object. Supported classes are
\code{strategic_predictor} and \code{conjoint_foundation_model}.}

\item{...}{Additional method-specific arguments.}

\item{newdata}{New data. For \code{strategic_predictor}, this follows the same
contract as \code{\link[stats]{predict}()}. For
\code{conjoint_foundation_model}, use either a list containing \code{W} and
optional \code{X}, \code{pair_id}, \code{profile_order}, and
\code{competing_group_variable_candidate},
\code{competing_group_variable_respondent}, and \code{experiment_id}, or a
data frame/matrix. When supplying a raw data frame to the foundation method,
extra numeric \code{X} columns can only be separated from factor columns when
\code{names_list} (or \code{p_list}) identifies the factor columns.}

\item{group_key}{Optional internal foundation group key to extract from.}

\item{mode}{Optional mode filter used for foundation-group selection.}

\item{likelihood}{Optional likelihood filter used for foundation-group
selection.}

\item{n_outcomes}{Optional outcome-count filter used for foundation-group
selection.}

\item{level}{Embedding level to return. \code{"candidate"} preserves the
historical candidate/profile output, \code{"respondent_context"} returns one
respondent-context matrix, and \code{"all"} returns both.}

\item{factor_schema}{Optional explicit prediction-time factor schema for fused
neural predictors. Supply a list with \code{names_list} or \code{p_list},
and optionally precomputed factor/level text or structural matrices.}

\item{text_embedding_fn}{Optional text embedding function used with
\code{factor_schema} and prediction-time \code{experiment_description}.}

\item{experiment_id}{Optional experiment identifier used for experiment-token
lookup.}

\item{names_list}{Optional local factor-level metadata for raw foundation
extraction.}

\item{p_list}{Optional \code{p_list} used to derive \code{names_list} when
needed.}

\item{canonical_factor_id}{Optional canonical factor ids used to map local
factors into a pooled foundation group.}

\item{canonical_level_id}{Optional canonical level ids used to map local levels
into a pooled foundation group.}

\item{foundation_adaptation_control}{Optional list controlling schema matching
for raw foundation extraction. Supported keys are
\code{strict_schema_match} and \code{allow_extra_covariates}.}

\item{conda_env}{Conda env name for neural extraction when params must be
reconstructed.}

\item{conda_env_required}{Require the conda env to exist.}
}
\value{
An object of class \code{strategic_embeddings}.

For \code{level = "candidate"}, single-mode models contain
\code{$embeddings}, an \eqn{n \times d} numeric matrix.

For pairwise models:
\itemize{
  \item non-\code{"full"} cross-candidate encoders return \code{$left} and
  \code{$right} matrices with one row per pair;
  \item \code{"full"} cross-candidate encoders return \code{$joint}, the
  pair-level final readout used immediately before the output head.
}

For \code{level = "respondent_context"}, the return value contains
\code{$respondent_context}, an \eqn{n \times d} or pair-level
\eqn{n_{pairs} \times d} matrix. For \code{level = "all"}, the candidate
fields and \code{$respondent_context} are returned together.

All outputs also include \code{$metadata}, including the source class, mode,
requested level, cross-candidate encoder mode, embedding width, and context
component availability.
}
\description{
Extract the final hidden representation immediately before the neural output
head, or the model's respondent-context representation.
}
\details{
\code{strategic_predictor} extraction works on fitted neural predictors,
including objects returned by \code{\link{strategic_prediction}()} and
\code{\link{adapt_conjoint_foundation_model}()}.

\code{conjoint_foundation_model} extraction runs directly on a selected pooled
foundation group. When extracting from a raw foundation object, local studies
must be mappable into the pooled schema, typically via matching
\code{experiment_id}, canonical ids, or direct use of pooled slot names and
level keys.

Respondent-context embeddings are derived from the model's context-side tokens:
respondent group, respondent covariates, experiment tokens, and pairwise
stage/matchup tokens when those components are available.

This function does not extend \code{predict()} or \code{predict_pair()}; it is
a dedicated embedding API.
}
\examples{
\dontrun{
fit <- strategic_prediction(
  Y = y,
  W = w,
  model = "neural",
  mode = "single"
)

emb <- extract_embeddings(fit, newdata = w[1:10, , drop = FALSE])

foundation_emb <- extract_embeddings(
  foundation_fit,
  newdata = list(
    W = target_w,
    X = target_x,
    pair_id = target_pair_id,
    profile_order = target_profile_order
  ),
  group_key = "pairwise::bernoulli::1",
  canonical_factor_id = c(price = "price", message = "message")
)
}
}
