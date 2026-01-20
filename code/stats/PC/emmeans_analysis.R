library(emmeans)
library(ggplot2)
library(brms)
library(bayestestR)
library(tidyverse)
library(tidybayes)
library(ggpubr)
library(car)
library(readxl)

dat_in <- # location of GLMM models #
ref_in <- # location of pc_scores_df.csv #

for (pc in 1:5){
  
  ref_f_name <- paste0(ref_in, "pc_scores_df.csv")
  
  #load data
  dat <- read_csv(ref_f_name, na="NA")
  
  #cond name
  nm <- paste0('PC',pc)
  
  # get dat SD vals for ROPE range 
  con_dat <- as.numeric(unlist(dat[nm]))
  
  rp_val <- 0.05 * sd(con_dat, na.rm=TRUE)
  
  setwd(paste0(dat_in, nm))
  
  #load brms fit
  fit <- readRDS(paste0(nm, '_fitObj.rds'))
  
  # #check chain convergence
  # plts <- plot(fit,
  #              variable = c("b_Intercept",
  #                           "b_cond1",
  #                           "b_time1",
  #                           "b_lobe1",
  #                           "b_lobe2",
  #                           "b_lobe3",
  #                           "b_lobe4",
  #                           "b_cond1:time1"),
  #              plot=FALSE)
  # 
  # conv_filename <- paste0(nm, "_conv_check.pdf")
  # pdf(conv_filename)
  
  # for (pt in 1:2) {
  #   print(plts[[pt]])
  # }
  # 
  # dev.off()

  # #posterior checks
  # for (par in c("cond", "time", "lobe")){
  #   fit_plt <- pp_check(fit, type='dens_overlay_grouped', group=par, ndraws = 100)
  #   
  #   pp_fname <- paste0(par, "_pp_check.jpeg")
  #   ggsave(pp_fname, plot=fit_plt, width=11, height=7, dpi=600)
  # }
  
  #get contrasts
  # emm_cnd <- emmeans(fit, pairwise ~ cond, level = 0.89, type = "response")
  # emm_tme <- emmeans(fit, pairwise ~ time, level = 0.89, type = "response")
  # emm_lbe <- emmeans(fit, pairwise ~ lobe, level = 0.89, type = "response")
  
  emm_condxtimexlobe <- emmeans(fit, pairwise ~ time|cond|lobe, level = 0.89, type = "response")
  emm_condxtime <- emmeans(fit, pairwise ~ time|cond, level = 0.89, type = "response")
  
  cont_list <- c("emm_cnd", "emm_tme", "emm_lbe", "emm_condxtimexlobe", "emm_condxtime")
  
  post_list <- list()
  emm_resp <- list()
  
  #comparisons between posteriors
  for(ii in 1:length(cont_list)){
    
    post_list[[length(post_list)+1]] <- describe_posterior(get(cont_list[ii])$contrasts,
                                                           ci = 0.89,
                                                           rope_range = c(-rp_val, rp_val),
                                                           rope_ci = 0.89,
                                                           ci_method = "HDI"
    )
    
    # save emm object
    saveRDS(get(cont_list[ii]), paste0(cont_list[ii], ".rds"))
    
    emm_resp[[length(emm_resp)+1]] <- as.data.frame(regrid(get(cont_list[ii])$emmeans))
    
  }
  # save posterior outputs
  saveRDS(post_list, 'postr_Description.rds')
  
  # save emmeans on response scale
  saveRDS(emm_resp, 'emm_respScale.rds')
}
