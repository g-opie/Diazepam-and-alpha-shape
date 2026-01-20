library(brms)
library(haven)
library (dplyr)
library(tidyverse)
library(readxl)
library(mice)
library(posterior)

mPath <- # root directory #
dat_path <- paste0(mPath, "group dat\\spect\\data\\")
out_path <- paste0(mPath, "stats\\spect\\models\\")

test <- c("a_frq", "a_pwr", "exp")

setwd(out_path)

#load data
file_name <- paste0(dat_path, "spect_dat.csv")
dat <- read_csv(file_name, na="NA")

#reclassify
dat$Grp <- as.factor(dat$Grp)
dat$Time <- as.factor(dat$Time)
dat$Subj <- as.factor(dat$Subj)

#loop PC
for(t in 1:length(test)){
  
  # create folder for test outputs
  subDir <- file.path(out_path, test[t])
  if(!dir.exists(subDir)){dir.create(subDir)}
  setwd(subDir)
  
  # get data as subset
  mod_dat <- dat[c("Subj","Grp", "Time", test[t])]

  # impute missing data
  m = 20
  imp_dat <- mice(mod_dat, m = m, print = FALSE)

  # save imputed data
  saveRDS(imp_dat, file = paste0(test[t], "_", "imp_dat.rds"))
  
  # load imputed data
  imp_dat <- readRDS(paste0(test[t], "_imp_dat.rds"))
  
  #run model
  fit <- brm_multiple(formula = brmsformula(paste0(test[t], '~ Grp * Time + (1 + Time | Subj)')),
                      data = imp_dat,
                      family = skew_normal(), 
                      warmup = 1000, 
                      iter = 4000, 
                      chains = 8, 
                      cores = 8, 
                      control = list(adapt_delta = 0.99, max_treedepth = 15))
  
  # # check convergence of sub-models
  draws <- as_draws_array(fit)
  draws_per_dat <- lapply(1:m, \(i) subset_draws(draws, variable="^b", chain = i, regex=TRUE))
  lapply(draws_per_dat, summarise_draws, default_convergence_measures())
  
  #save fit object
  saveRDS(fit, file = paste0(test[t], "_", "fitObj.rds"))
    
  }
