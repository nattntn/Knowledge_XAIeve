# =============================================================================
# Statistical validation for sex classification — Upper Third Molar (UTM)
# Tooth measurement: width (mesiodistal, mm)
# Random effect: random intercept per patient (accounts for pseudoreplication)
# =============================================================================

library(dplyr)
library(ggplot2)
library(lme4)       
library(lmerTest)   
library(performance) 
library(readr)

# =============================================================================
# 1. Load and prepare data
# =============================================================================
data <- read_csv("data/sex_upper_third_molar.csv")

# Factor encoding
data$patient_id <- as.factor(data$patient_id)
data$class      <- as.factor(data$class)   # "female" / "male"

cat("Observations :", nrow(data), "\n")
cat("Patients     :", n_distinct(data$patient_id), "\n")

# =============================================================================
# 2. Filter: keep only perturbation steps that changed model decision
#
# XI (X Increase) : prob_change > 0
# XD (X Decrease) : prob_change < 0
# =============================================================================
XI <- subset(data, prob_change > 0)
XD <- subset(data, prob_change < 0)
data_combined <- rbind(XI, XD)

cat("Observations after filter :", nrow(data_combined), "\n")
cat("Patients after filter     :", n_distinct(data_combined$patient_id), "\n")

# Verify no patient appears in both classes (should be 0)
data_combined %>%
  group_by(patient_id) %>%
  summarise(n_classes = n_distinct(class)) %>%
  filter(n_classes > 1) %>%
  nrow() %>%
  cat("Patients with both classes:", ., "\n")

# =============================================================================
# 3. Descriptive statistics
# =============================================================================
summary_table <- data_combined %>%
  group_by(class) %>%
  summarise(
    n_patients = n_distinct(patient_id),
    n_obs      = n(),
    mean       = round(mean(width_adj_mm), 2),
    sd         = round(sd(width_adj_mm), 2),
    min        = round(min(width_adj_mm), 2),
    max        = round(max(width_adj_mm), 2)
  )
print(summary_table)

# =============================================================================
# 4. Fit Linear Mixed-Effects Model (LMER via lme4)
#
# Fixed effect : class — sex (female / male)
# Random effect: random intercept per patient
# =============================================================================
model <- lmer(width_adj_mm ~ class + (1 | patient_id),
              data = data_combined)

summary(model)

# =============================================================================
# 5. Assumption checks
# =============================================================================
res <- resid(model)

# 5a. Normality of residuals — Q-Q plot
qqnorm(res, main = "Q-Q Plot: Residuals")
qqline(res)

# 5b. Normality of residuals — Histogram
hist(res, breaks = 30, main = "Histogram: Residuals",
     xlab = "Residuals")

# 5c. Homoscedasticity — Residuals vs Fitted
plot(fitted(model), res,
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, col = "red")

# 5d. Normality of random effects
ranef_vals <- ranef(model)$patient_id[, 1]
qqnorm(ranef_vals, main = "Q-Q Plot: Random Effects")
qqline(ranef_vals)

# =============================================================================
# 6. Inference
# =============================================================================

# Fixed-effects ANOVA table (Type III)
anova(model)

# Predicted marginal means per sex (population-level, re.form = NA)
newdata <- data.frame(class = c("female", "male"))
newdata$predicted <- predict(model, newdata, re.form = NA)
print(newdata)

# R-squared
r2(model)