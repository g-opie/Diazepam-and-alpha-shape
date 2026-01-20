library(brms)
library(tidyverse)

mPath <- # enter root path
dat_in <- paste0(mPath, "group dat\\emd\\")
dat_out <- paste0(mPath, "stats\\PCs\\models\\")
setwd(dat_in)

#load dat
dat <- read.csv("pc_scores_df.csv")

  #loop PC
for(pc in 2:5){
  
  # check for dat specific dir, create if not there, set wd
  subDir <- file.path(paste0(dat_out, 'PC', pc))
  if(!dir.exists(subDir)){dir.create(subDir)}
  setwd(subDir)

  #target PC
  pc_dat <- paste0("PC", pc)

  #remove nans
  mod_dat <- dat %>% drop_na(pc_dat)

  #reclassify
  mod_dat$cond <- as.factor(mod_dat$cond)
  mod_dat$time <- as.factor(mod_dat$time)
  mod_dat$sub <- as.factor(mod_dat$sub)
  mod_dat$lobe <- as.factor(mod_dat$lobe)
  mod_dat$ssd <- as.factor(mod_dat$ssd)

  mod_dat[[pc_dat]] <- as.numeric(mod_dat[[pc_dat]])

  #run model
  fit <- brm(formula = paste0('PC', pc, '~ cond * time * lobe + (1 + time + ssd|sub)'),
             data = mod_dat,
             prior = c(set_prior("normal(0,1)", class="b")),
             family = student(link = "identity"),
             warmup = 1000,
             iter = 4000,
             chains = 4,
             cores = 24,
             init = "0",
             control = list(adapt_delta = 0.99, max_treedepth = 15),
             threads = threading((6)),
             backend="cmdstanr")

  #save fit object
  saveRDS(fit, file = paste0('PC', pc, "_fitObj.rds"))
}
