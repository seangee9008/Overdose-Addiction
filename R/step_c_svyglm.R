# Cross-check for src/step_c.py: the same survey-weighted logistic models fitted with
# R survey::svyglm. Reads the extract written by step_c.py; writes outputs/step_c_r_check.csv.
#   Rscript R/step_c_svyglm.R
suppressPackageStartupMessages(library(survey))
d <- read.csv("data/raw/derived/nsduh_step_c_extract.csv")
for (v in c("YEAR", "CATAG3", "IRSEX", "NEWRACE2")) d[[v]] <- factor(d[[v]])
des <- svydesign(ids = ~VEREP, strata = ~VESTR_C, weights = ~ANALWT2_C1, data = d, nest = TRUE)
out <- list()
for (y in c("OPI_MISUSE", "OPI_OUD", "HERYR")) {
  for (m in c("crude", "adjusted")) {
    rhs <- if (m == "crude") "YEAR" else "YEAR + CATAG3 + IRSEX + NEWRACE2"
    fit <- svyglm(as.formula(paste(y, "~", rhs)), design = des, family = quasibinomial())
    cf <- summary(fit)$coefficients
    out[[length(out) + 1]] <- data.frame(outcome = y, model = m, term = rownames(cf),
                                         coef_r = cf[, 1], se_r = cf[, 2], df_r = fit$df.residual)
  }
}
res <- do.call(rbind, out)
res$term <- sub("^(YEAR|CATAG3|IRSEX|NEWRACE2)(.+)$", "\\1[\\2]", res$term)
write.csv(res, "outputs/step_c_r_check.csv", row.names = FALSE)
cat("survey", as.character(packageVersion("survey")), "- wrote outputs/step_c_r_check.csv\n")
