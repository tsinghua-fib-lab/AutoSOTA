
# Goods practices to follow

:warning:**You must never store credentials information into source code or config file in a GitHub repository**
- Block sensitive data being pushed to GitHub by git-secrets or its likes as a git pre-commit hook
- Audit for slipped secrets with dedicated tools
- Use environment variables for secrets in CI/CD (e.g. GitHub Secrets) and secret managers in production

# Security Policy

## Supported Versions

The lastest version of `treehfd` is currently being supported with security updates.

## Reporting a Vulnerability

If you believe you have found a security vulnerability, please let us know right away by contacting oss@thalesgroup.com.
We will investigate all legitimate reports and do our best to quickly fix the problem.

## Security Update policy

If security vulnerabilities are detected, we will track them below, and mitigate the problems in further releases.

## Known security gaps & future enhancements

None.
