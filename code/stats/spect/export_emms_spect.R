library(emmeans)
library(ggplot2)
library(brms)
library(bayestestR)
library(tidyverse)
library(tidybayes)
library(ggpubr)
library(car)

mPath <- # root directory #
dat_path <- paste0(mPath, "stats\\spect\\models\\")
out_path <- paste0(mPath, "stats\\spect\\models\\")

cond <- c("a_frq", "a_pwr", "exp") 

for (cnd in 1:length(cond)){
    
  setwd(paste0(out_path, cond[cnd]))
  
  #load and export EMMs
  emm <- readRDS(paste0(cond[cnd], "_emm.rds"))
  
  emm_out <- do.call(rbind, emm)
  
  write.csv(emm_out, file=paste0(cond[cnd], "_EMMs.csv"))
                       
  #load and export post description for all comparisons
  post <- readRDS(paste0(cond[cnd], "_postr_Description.rds"))
  
  post_out <- do.call(rbind, post)
  
  write.csv(post_out, file=paste0(cond[cnd], "_postr.csv"))
    
  
}