## Load packages ----
invisible({
  
  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "ggplot2", 
    "scales", "tictoc", "fable", "tsibble",
    "seasonal", "feasts"
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

## Load data & manipulate ----
train <- fread('raw-data/train.csv') %>% as_tibble()
test  <- fread('raw-data/test.csv') %>% as_tibble()

holidays_events <- fread('raw-data/holidays_events.csv') %>% as_tibble()
transactions    <- fread('raw-data/transactions.csv') %>% as_tibble()
oil_prices      <- fread('raw-data/oil.csv') %>% as_tibble()
stores          <- fread('raw-data/stores.csv') %>% as_tibble()

## 1a. EDA - Time series ----
train_ts <- train %>% 
  select(-id) %>% 
  as_tsibble(
    index = date,
    key = c(store_nbr, family)
  )

train_ts_agg <- train %>% 
  group_by(date, family) %>% 
  summarise(sales = sum(sales, na.rm = TRUE), .groups = 'drop') %>% 
  as_tsibble(
    index = date, key = family
  ) %>% 
  group_by_key() %>% 
  index_by(year_month = ~ yearmonth(.)) %>% 
  summarise(sales = sum(sales, na.rm = TRUE), .groups = 'drop')

train_ts %>% 
  index_by(vars(date, family)) %>% 
  summarise(sales = sum(sales, na.rm = TRUE))
  update_tsibble(key = c(family)) %>%
  group_by_key() %>% 
  summarise(sales = sum(sales, na.rm = TRUE)) %>% 
  # filter(family == 'AUTOMOTIVE') %>% 
  ggplot(aes(x = date, y = sales, colour = family)) +
  geom_line() +
  facet_wrap(vars(family), scales = 'free') +
  theme(legend.position = 'none')

train_ts %>% 
  group_by_key() %>% 
  index_by(year_month = ~ yearmonth(.)) %>% 
  summarise(sales = sum(sales, na.rm = TRUE), .groups = 'drop') %>% 
  ggplot(aes(x = year_month, y = sales, colour = family)) +
  facet_wrap(vars(family), scales = 'free') +
  geom_line() +
  theme(legend.position = 'none')

## 1b. EDA - Auxilliary datasets ----





## 
