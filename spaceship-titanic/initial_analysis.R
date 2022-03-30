## Load packages ----
invisible({

  # Packages to load are found in the character vector below
  packages <- c("tidyverse", "data.table", "ggplot2", "scales", "tictoc")

  if(!all(packages %in% rownames(installed.packages()))) {
  
    to_install <- packages[!(packages %in% rownames(installed.packages()))]
    lapply(to_install, install.packages, character.only = TRUE)
    rm(to_install)

  }

  lapply(packages, library, character.only = TRUE)
  rm(packages)

})

setwd('~/Documents/Projects/kaggle/spaceship-titanic')

## Load data & manipulate ----
train <- fread('raw-data/train.csv')
test  <- fread('raw-data/test.csv')

all   <- bind_rows(
  train %>% mutate(dataset = 'train'),
  test %>% mutate(dataset = 'test')
) 

## 1. EDA ----
