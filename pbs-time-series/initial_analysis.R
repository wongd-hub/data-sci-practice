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
  setwd('E:/Projects/kaggle-analyses/pbs-time-series') # PC
} else {
  setwd('~/Documents/Projects/kaggle/pbs-time-series') # MacOS
}

## Data Loads ----
dos <- fread('prepped_data/dos_final.csv')

dos_ts <- dos %>% 
  as_tsibble(index = month, key = c(pbs_item_code, starts_with('atc_level_')))

dos %>% 
  group_by(month, atc_level_1) %>% 
  summarise(rx = sum(prescriptions, na.rm = TRUE), .groups = 'drop') %>% 
  ggplot(aes(x = month, y = rx, colour = atc_level_1)) +
  geom_line()
