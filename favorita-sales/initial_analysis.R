## Load packages ----
invisible({
  
  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "ggplot2", 
    "scales", "tictoc", "infotheo", "ROCR",
    "Information", "gridExtra", "mice",
    "randomForest", "caret", "xgboost"
  )
  
  if(!all(packages %in% rownames(installed.packages()))) {
    
    to_install <- packages[!(packages %in% rownames(installed.packages()))]
    lapply(to_install, install.packages, character.only = TRUE)
    rm(to_install)
    
  }
  
  lapply(packages, library, character.only = TRUE)
  rm(packages)
  
})

if (.Platform$OS.type == "windows") {
  setwd('E:/Projects/kaggle-analyses/favorita-sales') # PC
} else {
  setwd('~/Documents/Projects/kaggle/favorita-sales') # Macbook
}