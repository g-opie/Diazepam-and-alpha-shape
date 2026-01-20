library(emmeans)
library(ggplot2)
library(brms)
library(bayestestR)
library(tidyverse)
library(tidybayes)
library(ggpubr)
library(car)

dat_in <- # location of EM means data #

for (pc in 1:5){
    
  setwd(paste0(dat_in, "PC", pc, "\\"))
  
  #load and export EMMs
  emm <- readRDS("emm_condxtimexlobe.rds")
  write.csv(emm$emmeans, file=paste0("PC", pc, '_EMMs_3int.csv'))
  
  emm <- readRDS("emm_condxtime.rds")
  write.csv(emm$emmeans, file=paste0("PC", pc, '_EMMs_2int.csv'))
                       
  #load and export post description for all comparisons
  post <- readRDS("postr_Description.rds")
  post_out <- do.call(rbind, post)
  write.csv(post_out, file=paste0("PC", pc, "_posterior_out.csv"))
}