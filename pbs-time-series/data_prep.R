## Purpose ----
# The purpose of this script is to process PBS Date of Supply dispense volume
# data and join it up to information provided in the PBS pricing files to get
# the Anatomical Therapeutic Chemical (ATC) code for each PBS item.

## Load packages ----
invisible({
  
  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "rvest", "lubridate"
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
  setwd('~/Documents/Projects/kaggle/pbs-time-series') # Macbook
}

## Process Date of Supply dispense volumes data ----
dos <- fread('raw_data/dos-jul-2017-to-feb-2022.csv')

# Aggregate to the Item Code and Month level
dos_wrk <- dos %>% 
  group_by(MONTH_OF_SUPPLY, ITEM_CODE) %>% 
  summarise(PRESCRIPTIONS = sum(PRESCRIPTIONS, na.rm = TRUE), .groups = 'drop') %>% 
  mutate(MONTH_OF_SUPPLY = ymd(paste0(MONTH_OF_SUPPLY, '01')))

# Grab distinct list of months to use in scraping
distinct_months <- dos %>% 
  distinct(MONTH_OF_SUPPLY) %>% 
  mutate(MONTH_OF_SUPPLY = ymd(paste0(MONTH_OF_SUPPLY, '01')))

## Downloading Pricing files from the PBS Publications Archive ----
# Extract HTML from Publications Archive page
pbs_publications_archive <- read_html('https://www.pbs.gov.au/info/publication/schedule/archive')

# Retrieve all <a> elements with a title that starts with 'PBS Text files (ZIP',
# then extract link attribute
raw_dl_list <- pbs_publications_archive %>% 
  html_nodes('a[title^="PBS Text files (ZIP"]') %>% 
  html_attr('href')

# Filter list of links to only months we've got Date of Supply data for Where
# there are duplicates, choose the last link, assuming that'd be the most
# updated
dl_list <- tibble(link = test) %>% 
  mutate(scraped_date = str_extract(link, '[0-9]{4}-[0-9]{2}-01') %>% ymd()) %>% 
  inner_join(distinct_months, by = c('scraped_date' = 'MONTH_OF_SUPPLY')) %>% 
  group_by(scraped_date) %>% 
  summarise(first_link = last(link), .groups = 'drop') %>% 
  # Fix relative links
  mutate(first_link = str_replace(first_link, '/publication/schedule/../../', 'https://www.pbs.gov.au/'))

# Initialise results list
schedule_dl <- vector('list', length(dl_list))

# Older text files don't have column names included; so pull column names from a
# file that definitely has them and use it for those
tmp_file <- tempfile()
tmp_sched_month <- max(dl_list$scraped_date)

download.file(
  dl_list %>% 
    dplyr::filter(scraped_date == tmp_sched_month) %>% 
    pull(first_link), 
  tmp_file
)

col_names <- read_delim(
  unz(
    tmp_file, 
    paste0(
      'drug_', 
      str_remove_all(tmp_sched_month, '-'), 
      '.txt'
    )
  ), 
  col_names = TRUE,
  delim = '!'
) %>% 
  colnames()

unlink(tmp_file)

# Loop through list of downloads, downloading and unzipping into raw_data/pricing
for (i in 1:nrow(dl_list)) {
  
  # Pull date and link for iteration
  schedule_month <- dl_list[i, 'scraped_date'] %>% pull(scraped_date)
  schedule_link  <- dl_list[i, 'first_link'] %>% pull(first_link)
  
  # Set up skeleton temp file to store downloaded table in
  tmp_file <- tempfile()
  
  data <- tryCatch(
    {

      print(paste0('Downloading schedule for', schedule_month))
      download.file(schedule_link, tmp_file)

      print('Checking column names')
      tmp_col_names <- read_delim(
        unz(
          tmp_file, 
          paste0(
            'drug_', 
            str_remove_all(schedule_month, '-'), 
            '.txt'
          )
        ), 
        delim = '!'
      ) %>% 
        colnames()
      
      if (tmp_col_names[1] != 'program-code') {
        
        print('Correct column names not present, replacing with proper column names')
        data <- read_delim(
          unz(
            tmp_file, 
            paste0(
              'drug_', 
              str_remove_all(schedule_month, '-'), 
              '.txt'
            )
          ), 
          col_names = col_names,
          delim = '!'
        ) %>% 
          mutate(date = schedule_month)
        
      } else {
        
        print('Correct column names present')
        data <- read_delim(
          unz(
            tmp_file, 
            paste0(
              'drug_', 
              str_remove_all(schedule_month, '-'), 
              '.txt'
            )
          ), 
          col_names = TRUE,
          delim = '!'
        ) %>% 
          mutate(date = schedule_month)
        
      }
      
    },
    error = function(cond) {
      warning_msg <- paste('Download for', schedule_month, 'has failed.', 'Original error message:', cond)
      warning(warning_msg)
      return(NA)
    },
    
    # Purge temp file in preparation for next download
    finally = {unlink(tmp_file)}
  )

  print('Saving results of iteration')
  schedule_dl[[i]] <- data
  
  # Sleep for 10 seconds to avoid server/download issues
  print('Timing out for 10 seconds to avoid server issues...')
  Sys.sleep(10)
  
}
  
# Bind each month into a single data-frame and save
full_schedule <- schedule_dl %>% rbindlist()
fwrite(full_schedule, 'prepped_data/full_schedule.csv')

## Final Setup ----
# Extract ATC information for each PBS code. Some PBS codes are listed under more
# than 1 ATC code, a simplifying assumption has been made that assumes the PBS
# code's dispenses are split evenly across those shared ATC codes.
item_code_to_atc_wrk <- full_schedule %>% 
  select(date, atc, `item-code`) %>% 
  distinct()

item_code_to_atc <- item_code_to_atc_wrk %>% 
  left_join(
    item_code_to_atc_wrk %>% 
      group_by(date, `item-code`) %>% 
      summarise(n_atcs = n(), .groups = 'drop'),
    by = c('date', 'item-code')
  ) %>% 
  mutate(atc_wgt = 1 / n_atcs, .keep = 'unused')

# Add ATC code information on to dos_wrk
dos_w_atc <- dos_wrk %>% 
  left_join(item_code_to_atc, by = c('MONTH_OF_SUPPLY' = 'date', 'ITEM_CODE' = 'item-code')) %>% 
  # Note items such as Extemporaneous Items or Standard Formula Preparations do not have ATC codes
  dplyr::filter(!is.na(atc))

# Split prescriptions across duplicate ATC codes and split ATC codes into their
# 5 levels
dos_final <- dos_w_atc %>% 
  mutate(
    PRESCRIPTIONS = round(PRESCRIPTIONS * atc_wgt), 
    atc_level_1 = str_sub(atc, 1, 1), # First level of ATC: Anatomical Main Group
    atc_level_2 = str_sub(atc, 1, 3), # Second: Therapeutic Subgroup
    atc_level_3 = str_sub(atc, 1, 4), # Third : Therapeutic/Pharmacological Subgroup 
    atc_level_4 = str_sub(atc, 1, 5), # Fourth: Chemical/Therapeutic/Pharmacological Subgroup
    atc_level_5 = atc,                # Fifth : Chemical Substance
    .keep = 'unused'
  ) %>% 
  rename(
    month         = MONTH_OF_SUPPLY,
    pbs_item_code = ITEM_CODE,
    prescriptions = PRESCRIPTIONS
  )

fwrite(dos_final, 'prepped_data/dos_final.csv')
