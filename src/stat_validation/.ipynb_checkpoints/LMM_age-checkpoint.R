# =============================================================================
# lmm_age.R
# -----------------------------------------------------------------------------
# Statistical validation for the age estimation task
# Model: Linear Mixed-Effects Model (LME) with quadratic crown-root ratio term
# Random effect: random intercept per patient (accounts for pseudoreplication)
# =============================================================================

library(dplyr)
library(nlme)       
library(performance) 
library(car)        
library(ggplot2)


# =============================================================================
# 1. Load and prepare data
# =============================================================================
data <- read.csv("data/age.csv")

# Factor encoding
data$patient_id          <- as.factor(data$patient_id)
data$Gender_predict_int  <- as.factor(data$Gender_predict_int)

# Filter: restrict to developmental age window and remove outliers
data <- subset(data, Age_predict_int >= 10 & Age_predict_int <= 25)
data <- subset(data, ratio_cr <= 5)

# Mean-centre crown-root ratio to reduce collinearity with quadratic term
data$ratio_cr_c <- data$ratio_cr - mean(data$ratio_cr)

cat("Observations :", nrow(data), "\n")
cat("Patients     :", n_distinct(data$patient_id), "\n")


# =============================================================================
# 2. Patient-level correlation (Pearson)
# Aggregate to patient level first to avoid pseudoreplication in correlation
# =============================================================================
data_agg <- data %>%
  group_by(patient_id) %>%
  summarise(
    mean_ratio_cr     = mean(ratio_cr),
    mean_age_predict  = mean(Age_predict)
  )

cor.test(data_agg$mean_ratio_cr,
         data_agg$mean_age_predict,
         method = "pearson")


# =============================================================================
# 3. Fit Linear Mixed-Effects Model (LME via nlme)
#
# Fixed effects:
#   ratio_cr_c             — linear crown-root ratio (mean-centred)
#   I(ratio_cr_c^2)        — quadratic term (U-shaped developmental curve)
#   Gender_predict_int     — sex (0 = Female, 1 = Male)
#   ratio_cr_c:Gender_predict_int — interaction: sex modifies ratio effect
#
# Random effect: random intercept per patient
#   accounts for repeated measures (multiple perturbation steps per patient)
#
# Variance structure: varPower
#   allows residual variance to scale with crown-root ratio
# =============================================================================
model <- lme(
  Age_predict ~ ratio_cr_c + I(ratio_cr_c^2) +
    Gender_predict_int +
    ratio_cr_c:Gender_predict_int,
  random  = ~ 1 | patient_id,
  weights = varPower(form = ~ ratio_cr_c),
  data    = data
)

summary(model)

# =============================================================================
# 4. Assumption checks
# =============================================================================

res <- resid(model)

# 4a. Residuals vs Fitted ( Homoscedasticity)
plot(fitted(model), res,
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, col = "red")

# 4b. Q-Q plot of residuals (Normality of residuals)
qqnorm(res, main = "Q-Q Plot: Residuals")
qqline(res)

# 4c. Q-Q plot of random effects (Normality of random effects)
ranef_vals <- random.effects(model)$`(Intercept)`
qqnorm(ranef_vals, main = "Q-Q Plot: Random Effects")
qqline(ranef_vals)

# 4d. VIF (from equivalent fixed-effects model) (Multicollinearity)
model_lm <- lm(
  Age_predict ~ ratio_cr_c + I(ratio_cr_c^2) +
    Gender_predict_int +
    ratio_cr_c:Gender_predict_int,
  data = data
)
vif(model_lm) 


# =============================================================================
# 5. Model fit: R-squared and ICC
# =============================================================================
r2(model)


# =============================================================================
# 6. Predict over grid (fixed effects only, level = 0)
# =============================================================================
newdata <- expand.grid(
  ratio_cr_c         = seq(min(data$ratio_cr_c),
                            max(data$ratio_cr_c),
                            length.out = 500),
  Gender_predict_int = factor(c(0, 1))
)

newdata$ratio_cr    <- newdata$ratio_cr_c + mean(data$ratio_cr)
newdata$Age_predict <- predict(model, newdata, level = 0)
newdata$Sex         <- ifelse(newdata$Gender_predict_int == 0,
                               "Female (0)", "Male (1)")

# Minimum predicted age per sex
min_point <- newdata %>%
  group_by(Sex) %>%
  slice_min(Age_predict) %>%
  slice(1)

print(min_point)


# =============================================================================
# 7. Plots
# =============================================================================

# 7a. Patient-level scatter with LOESS
ggplot(data_agg, aes(x = mean_ratio_cr, y = mean_age_predict)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "loess", color = "red", se = TRUE) +
  labs(x = "Crown-Root Ratio",
       y = "Predicted age (years)") +
  theme_bw()

# 7b. LMM fixed-effects curve with minimum points
ggplot(newdata, aes(x = ratio_cr, y = Age_predict,
                    color = Sex, linetype = Sex)) +
  geom_line(size = 1) +
  geom_point(data = min_point,
             aes(x = ratio_cr, y = Age_predict),
             size = 3,
             color = c("#C20000", "#1A5C99")) +
  geom_text(data = min_point,
            aes(label = paste0("Min (", round(ratio_cr, 2),
                               ", ", round(Age_predict, 2), ")")),
            hjust = -0.1, size = 4) +
  scale_color_manual(values = c("Female (0)" = "pink",
                                 "Male (1)"   = "lightblue")) +
  scale_linetype_manual(values = c("Female (0)" = "solid",
                                    "Male (1)"   = "dashed")) +
  labs(x = "Crown-Root Ratio",
       y = "Predicted age (years)",
       color = "Sex", linetype = "Sex") +
  theme_bw()