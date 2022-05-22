## Load packages ----
invisible({
  
  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "ggplot2", 
    "scales", "tictoc", "gridExtra", "fable",
    "seasonal", "feasts", "fGarch"
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
  setwd('E:/Projects/kaggle-analyses/stock-forecasting') # PC
} else {
  setwd('~/Documents/Projects/kaggle/stock-forecasting') # Mac
}

## Introduction ----
# This is an attempt at modelling the returns of Microsoft and Apple stocks using ARIMA/ARIMA-GARCH methods.
# Data is sourced from https://www.kaggle.com/datasets/paultimothymooney/stock-market-data as at 20 May 2022.

## Load data & manipulate ----
aapl <- fread('raw-data/AAPL.csv') %>% as_tibble() %>% mutate(Date = lubridate::dmy(Date))
msft <- fread('raw-data/MSFT.csv') %>% as_tibble() %>% mutate(Date = lubridate::dmy(Date))

all_df <- bind_rows(
  aapl %>% mutate(ticker = 'AAPL'),
  msft %>% mutate(ticker = 'MSFT')
) 

## 1a. EDA - Categorical Variables ----