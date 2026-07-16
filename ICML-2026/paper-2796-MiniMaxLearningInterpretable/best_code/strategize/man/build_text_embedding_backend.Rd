\name{build_text_embedding_backend}
\alias{build_text_embedding_backend}
\title{Build a host-aware text-embedding backend}
\usage{
build_text_embedding_backend(
  conda_env = "strategize_torch_env",
  family = NULL,
  runtime = "auto",
  profile = "portable",
  model_id = NULL,
  cache_dir = NULL,
  cache_only = FALSE,
  batch_size = 64L,
  required = TRUE,
  verbose = TRUE,
  conda = "auto"
)
}
\arguments{
\item{conda_env}{Conda env name used for the Python runtime. Defaults to
\code{"strategize_torch_env"} so text embedding packages can be isolated
from the JAX backend environment.}

\item{family}{Optional embedding-family selector. When \code{NULL}, the
selected profile supplies the family.}

\item{runtime}{Runtime preference. Use \code{"auto"} to resolve per host, or choose one of \code{"mlx"}, \code{"cuda"}, \code{"rocm"}, or \code{"cpu"} explicitly.}

\item{profile}{Embedding profile. Use \code{"portable"} for the default
portable profile or a registered profile such as
\code{"harrier_oss_v1_0.6b_1024"} or \code{"qwen3_8b_4096"}.}

\item{model_id}{Optional model id override applied to every compatible
candidate in the selected profile.}

\item{cache_dir}{Optional directory for cached text embeddings.}

\item{cache_only}{Logical; if \code{TRUE}, missing cache entries error before
a model runtime is imported.}

\item{batch_size}{Number of texts to encode per model call.}

\item{required}{Logical; if \code{TRUE}, ensure the selected runtime is usable immediately. If \code{FALSE}, return a lazy backend that validates on first use.}

\item{verbose}{Logical; if \code{TRUE}, print install milestones and stream
long-running pip/conda output.}

\item{conda}{Conda binary to use. Defaults to \code{"auto"}.}
}
\value{
A list with \code{$fn}, \code{$spec}, and \code{$runtime}.
}
\description{
Build a host-aware text-embedding backend
}
\examples{
\dontrun{
backend <- build_text_embedding_backend(conda_env = "strategize_torch_env")

study_a$experiment_description <- "Source study A"
study_b$experiment_description <- "Source study B"

foundation_fit <- preference.fm::fit_conjoint_foundation_model(
  experiments = list(study_a, study_b),
  foundation_control = list(
    text_embedding_fn = backend$fn
  )
)

inspect_text_embedding_backend(conda_env = "strategize_torch_env")
}
}
