# =============================================================================
# Statistical validation for sex classification — Upper Third Molar
# Validation dataset (original tooth measurements)
#
# Two dimensions are analysed independently:
#   MD  : Mesiodistal width  (width_mm)  — Section A
#   TL  : Total length/height (height_mm) — Section B
# =============================================================================

library(dplyr)
library(ggplot2)
library(lme4)        
library(lmerTest)    
library(performance)
library(readr)

# =============================================================================
# Age sub-groups available in this dataset:
#   Group 1 : Age 14–17  → change filter below to >= 14 & <= 17
#   Group 2 : Age 18–25  → change filter below to >= 18 & <= 25
#   All     : Age 14–25  → current default
# =============================================================================
data_all <- read_csv("data/sex_upper_third_molar_validation.csv")

# Age filter — adjust bounds to select sub-group
data <- subset(data_all, `Age(year)` >= 14 & `Age(year)` <= 25)

# Factor encoding
data$patient_id <- as.factor(data$patient_id)
data$Gender     <- as.factor(data$Gender)   # "F" / "M"

cat("Observations :", nrow(data), "\n")
cat("Patients     :", n_distinct(data$patient_id), "\n")

# Check for patients with inconsistent sex labels
patient_sex <- data %>%
  group_by(patient_id) %>%
  summarise(n_sex     = n_distinct(Gender),
            sex_values = paste(unique(Gender), collapse = "/")) %>%
  filter(n_sex > 1)

cat("Patients with conflicting sex labels:", nrow(patient_sex), "\n")
if (nrow(patient_sex) > 0) print(patient_sex)

# Correlation between MD and TL (check before running each section)
cat("Correlation MD vs TL:", cor(data$width_mm, data$height_mm), "\n")

# =============================================================================
# SECTION A — Mesiodistal width (MD)
# Outcome variable: width_mm
# =============================================================================

# 3A. Descriptive statistics
summary_table_MD <- data %>%
  group_by(Gender) %>%
  summarise(
    n_patients = n_distinct(patient_id),
    n_obs      = n(),
    mean       = round(mean(width_mm), 2),
    sd         = round(sd(width_mm), 2),
    min        = round(min(width_mm), 2),
    max        = round(max(width_mm), 2)
  )
print(summary_table_MD)

# 4A. Fit LMM — MD
model_MD <- lmer(width_mm ~ Gender + (1 | patient_id),
                 data = data)
summary(model_MD)

# 5A. Assumption checks — MD
res_MD <- resid(model_MD)

qqnorm(res_MD, main = "Q-Q Plot: Residuals (MD)")
qqline(res_MD)

hist(res_MD, breaks = 30, main = "Histogram: Residuals (MD)",
     xlab = "Residuals")

plot(fitted(model_MD), res_MD,
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted (MD)")
abline(h = 0, col = "red")

ranef_MD <- ranef(model_MD)$patient_id[, 1]
qqnorm(ranef_MD, main = "Q-Q Plot: Random Effects (MD)")
qqline(ranef_MD)

# 6A. Inference — MD
anova(model_MD)

newdata_MD <- data.frame(Gender = c("F", "M"))
newdata_MD$predicted <- predict(model_MD, newdata_MD, re.form = NA)
print(newdata_MD)

r2(model_MD)

# =============================================================================
# SECTION B — Total length / height (TL)
# Outcome variable: height_mm
# =============================================================================

# 3B. Descriptive statistics
summary_table_TL <- data %>%
  group_by(Gender) %>%
  summarise(
    n_patients = n_distinct(patient_id),
    n_obs      = n(),
    mean       = round(mean(height_mm), 2),
    sd         = round(sd(height_mm), 2),
    min        = round(min(height_mm), 2),
    max        = round(max(height_mm), 2)
  )
print(summary_table_TL)

# 4B. Fit LMM — TL
model_TL <- lmer(height_mm ~ Gender + (1 | patient_id),
                 data = data)
summary(model_TL)

# 5B. Assumption checks — TL
res_TL <- resid(model_TL)

qqnorm(res_TL, main = "Q-Q Plot: Residuals (TL)")
qqline(res_TL)

hist(res_TL, breaks = 30, main = "Histogram: Residuals (TL)",
     xlab = "Residuals")

plot(fitted(model_TL), res_TL,
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted (TL)")
abline(h = 0, col = "red")

ranef_TL <- ranef(model_TL)$patient_id[, 1]
qqnorm(ranef_TL, main = "Q-Q Plot: Random Effects (TL)")
qqline(ranef_TL)

# 6B. Inference — TL
anova(model_TL)

newdata_TL <- data.frame(Gender = c("F", "M"))
newdata_TL$predicted <- predict(model_TL, newdata_TL, re.form = NA)
print(newdata_TL)

r2(model_TL)