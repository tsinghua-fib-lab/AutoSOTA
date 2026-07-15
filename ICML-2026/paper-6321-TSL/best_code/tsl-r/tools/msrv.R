# read the DESCRIPTION file
desc <- read.dcf("DESCRIPTION")

if (!"SystemRequirements" %in% colnames(desc)) {
  fmt <- c(
    "`SystemRequirements` not found in `DESCRIPTION`.",
    "Please specify `SystemRequirements: Cargo (Rust's package manager), rustc`"
  )
  stop(paste(fmt, collapse = "\n"))
}

# extract system requirements
sysreqs <- desc[, "SystemRequirements"]

# check that cargo and rustc is found
if (!grepl("cargo", sysreqs, ignore.case = TRUE)) {
  stop("You must specify `Cargo (Rust's package manager)` in your `SystemRequirements`")
}

if (!grepl("rustc", sysreqs, ignore.case = TRUE)) {
  stop("You must specify `Cargo (Rust's package manager), rustc` in your `SystemRequirements`")
}

# split into parts
parts <- strsplit(sysreqs, ", ")[[1]]

# identify which is the rustc
rustc_ver <- parts[grepl("rustc", parts)]

# perform checks for the presence of rustc and cargo on the OS
no_cargo_msg <- c(
  "----------------------- [CARGO NOT FOUND]--------------------------",
  "The 'cargo' command was not found on the PATH. Please install Cargo",
  "from: https://www.rust-lang.org/tools/install",
  "",
  "Alternatively, you may install Cargo from your OS package manager:",
  " - Debian/Ubuntu: apt-get install cargo",
  " - Fedora/CentOS: dnf install cargo",
  " - macOS: brew install rust",
  "-------------------------------------------------------------------"
)

no_rustc_msg <- c(
  "----------------------- [RUST NOT FOUND]---------------------------",
  "The 'rustc' compiler was not found on the PATH. Please install",
  paste(rustc_ver, "or higher from:"),
  "https://www.rust-lang.org/tools/install",
  "",
  "Alternatively, you may install Rust from your OS package manager:",
  " - Debian/Ubuntu: apt-get install rustc",
  " - Fedora/CentOS: dnf install rustc",
  " - macOS: brew install rust",
  "-------------------------------------------------------------------"
)

# Add {user}/.cargo/bin to path before checking
new_path <- paste0(
  Sys.getenv("PATH"),
  ":",
  paste0(Sys.getenv("HOME"), "/.cargo/bin")
)

# set the path with the new path
Sys.setenv("PATH" = new_path)

# check for rustc installation
rustc_version <- tryCatch(
  system("rustc --version", intern = TRUE),
  error = function(e) {
    stop(paste(no_rustc_msg, collapse = "\n"))
  }
)

# check for cargo installation
cargo_version <- tryCatch(
  system("cargo --version", intern = TRUE),
  error = function(e) {
    stop(paste(no_cargo_msg, collapse = "\n"))
  }
)

# helper function to extract versions
extract_semver <- function(ver) {
  if (grepl("\\d+\\.\\d+(\\.\\d+)?", ver)) {
    sub(".*?(\\d+\\.\\d+(\\.\\d+)?).*", "\\1", ver)
  } else {
    NA
  }
}

# get the MSRV
msrv <- extract_semver(rustc_ver)

# extract current version
current_rust_version <- extract_semver(rustc_version)

# perform check
if (!is.na(msrv)) {
  # -1 when current version is later
  # 0 when they are the same
  # 1 when MSRV is newer than current
  is_msrv <- utils::compareVersion(msrv, current_rust_version)
  if (is_msrv == 1) {
    fmt <- paste0(
      "\n------------------ [UNSUPPORTED RUST VERSION]------------------\n",
      "- Minimum supported Rust version is %s.\n",
      "- Installed Rust version is %s.\n",
      "---------------------------------------------------------------"
    )
    stop(sprintf(fmt, msrv, current_rust_version))
  }
}

# print the versions
versions_fmt <- "Using %s\nUsing %s"
message(sprintf(versions_fmt, cargo_version, rustc_version))
