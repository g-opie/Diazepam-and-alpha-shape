library(emmeans)
library(ggplot2)
library(brms)
library(bayestestR)
library(tidyverse)
library(tidybayes)
library(ggpubr)
library(car)

mPath <- # root directory #
dat_path <- paste0(mPath, "group dat\\spect\\data\\")
out_path <- paste0(mPath, "stats\\spect\\models\\")

files <- c("a_frq", "a_pwr", "exp") 

#load data
file_name <- paste0(dat_path, "spect_dat.csv")
dat <- read_csv(file_name, na="NA")

for(i in 1:length(files)) {
  
  # get vals for rope (10% SD)
  con_dat <- as.numeric(unlist(dat[files[i]]))
  rp_val <- 0.1 * sd(con_dat, na.rm=TRUE)
  
  # check for dat specific dir, create if not there, set wd
  subDir <- file.path(out_path, files[i])
  if(!dir.exists(subDir)){dir.create(subDir)}
  setwd(subDir)
  
  #load brms fit
  fit <- readRDS(paste0(files[i], "_fitObj.rds"))

  #posterior checks
  fit_check_Grp <- pp_check(fit, type='dens_overlay_grouped', group='Grp', ndraws = 100)
  fit_check_Time <- pp_check(fit, type='dens_overlay_grouped', group='Time', ndraws = 100)
  mrg_plot <- ggarrange(fit_check_Grp, fit_check_Time, ncol=1, nrow=2)

  pp_fname <- paste0(files[i], "_pp_check.jpeg")
  ggsave(pp_fname, plot=mrg_plot)
  
  #get contrasts
  emm_grp <- emmeans(fit, pairwise ~ Grp, level = 0.95)
  emm_time <- emmeans(fit, pairwise ~ Time, level = 0.95)
  emm_int <- emmeans(fit, pairwise ~ Grp*Time, level = 0.95)
  
  cont_list <- c("emm_grp", "emm_time", "emm_int")
  
  post_list <- list()
  emm_resp <- list()
  
  #comparisons between posteriors
  for(ii in 1:length(cont_list)){
    
    post_list[[length(post_list)+1]] <- describe_posterior(get(cont_list[ii])$contrasts,
                                                           ci = 0.95,
                                                           rope_range = c(-(rp_val), rp_val),
                                                           rope_ci = 0.95,
                                                           ci_method = "HDI")
    
    # save emm object
    saveRDS(get(cont_list[ii]), paste0(files[i], "_", cont_list[ii], ".rds"))
    
    emm_resp[[length(emm_resp)+1]] <- as.data.frame(regrid(get(cont_list[ii])$emmeans))
    
  }
  # save posterior outputs
  saveRDS(post_list, paste0(files[i], '_postr_Description.rds'))
  
  # save emmeans on response scale
  saveRDS(emm_resp, paste0(files[i], '_emm.rds'))
}
