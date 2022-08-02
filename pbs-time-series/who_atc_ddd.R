## Purpose ----
# The purpose of this script is to scrape Defined Daily Doses for a select set of 
# Anatomical Therapeutic Classification codes.

## Load packages ----
invisible({
  
  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "rvest"
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

## Build scraping function ----

atc_regex <- list(
  atc1 = r'([A-Z]{1})',
  atc2 = r'([A-Z]{1}[0-9]{2})',
  atc3 = r'([A-Z]{1}[0-9]{2}[A-Z]{1})',
  atc4 = r'([A-Z]{1}[0-9]{2}[A-Z]{1}[A-Z]{1})',
  atc5 = r'([A-Z]{1}[0-9]{2}[A-Z]{1}[A-Z]{1}[0-9]{2})'
)

get_child_atcs <- function(atc_code, main_link = 'https://www.whocc.no/atc_ddd_index') {
  
  # Lookup regex rules for this ATC code level and the next, as well as the difference
  #  (increment) between the two for use in figuring out which links point to the
  #  children of atc_code
  current_level_regex <- atc_regex[
    lapply(atc_regex, function(x) str_detect(atc_code, x)) %>% unlist()
    ] %>% 
    tail(1)

  current_level_num <- names(current_level_regex)[[1]] %>% 
    str_extract(r'([1-5]{1})') %>% 
    as.integer()

  incremental_regex <- str_remove(
    atc_regex[[str_interp('atc${current_level_num + 1}')]], 
    fixed(current_level_regex[[1]])
  )

  # Read HTML from target site for given ATC code; harvest all links
  tmp_url <- url(str_interp('${main_link}/?code=${atc_code}&showdescription=no'), 'rb')
  atc_html_tmp <- read_html(tmp_url)
  close(tmp_url)
  
  atc_list_tmp <- atc_html_tmp %>% 
    html_elements('#content') %>% 
    html_elements('a') %>% 
    html_attr('href')

  # Filter and process list of links to only those that are children of atc_code
  filtered_atc_list_tmp <- atc_list_tmp[
    # This filters the list of available links down to those that are 
    #  the children of the ATC code passed to get_child_atcs()
    str_detect(atc_list_tmp, paste0(r'[./?code=]', atc_code, incremental_regex))
  ] %>% 
    # Converting relative links to absolute
    str_replace(pattern = r'[.]', main_link) %>%
    # Enforce showdescription = no
    str_replace(pattern = r'[yes$]', replacement = 'no')
  
  return(filtered_atc_list_tmp)
  
}

check_and_harvest_table <- function(link) {
  
  tmp_url <- url(link, 'rb')
  
  tbl_tmp <- read_html(tmp_url) %>% 
    html_elements('#content') %>% 
    html_elements('table')
  
  close(tmp_url)
  
  if (length(tbl_tmp) == 0) {
    
    return(tibble())
    
  } else {

    html_tbl_tmp <- tbl_tmp %>% 
      html_table(header = TRUE, fill = TRUE) %>% 
      bind_rows() %>% 
      mutate_all(.funs = ~ na_if(., '')) %>%
      fill(`ATC code`, Name, .direction = 'down')
    
    # Enforce column types
    html_tbl <- html_tbl_tmp %>% 
      mutate_at(
        .vars = vars(`ATC code`, Name, U, Adm.R, Note), 
        .funs = as.character
      ) %>% 
      mutate_at(
        .vars = vars(DDD),
        .funs = as.double
      )
    
    return(html_tbl)
    
  }
  
}

scrape_ddds_atc2 <- function(
  atc2s, main_link = 'https://www.whocc.no/atc_ddd_index'
) {
  
  # This function takes an ATC2 level code and walks through
  #  all children links underneath it through to the terminal
  #  node (ATC5, or the earliest ATC level that has a table present).
  #  At each terminal node, the DDD table will be harvested and added
  #  to the results dataframe.
  
  # atc2s takes either a single string (e.g. 'N05') or a character
  #  vector (e.g. c('N05', 'J01')).
  
  atc5_ddds <- tibble(
    `ATC code` = character(),
    Name       = character(),
    DDD        = double(),
    U          = character(),
    Adm.R      = character(),
    Note       = character()
  )
  
  for (atc2 in seq_along(atc2s)) {
    
    writeLines(str_interp('\nProcessing ${atc2s[[atc2]]} (${atc2}/${length(atc2s)}):'))
    
    tbl_tmp <- check_and_harvest_table(str_interp('${main_link}/?code=${atc2s[[atc2]]}&showdescription=no'))
    
    if (nrow(tbl_tmp) > 0) {
      
      writeLines(str_interp('└ Found non-zero table at ${atc2s[[atc2]]}, scraping...'))
      atc5_ddds <- bind_rows(atc5_ddds, tbl_tmp)
      rm(tbl_tmp)
      
      next
    }

    filtered_atc3_list <- get_child_atcs(atc2s[[atc2]], main_link)

    for (atc3_link in seq_along(filtered_atc3_list)) {

      atc3_tmp <- str_extract(filtered_atc3_list[atc3_link], atc_regex$atc3)

      writeLines(str_interp('└ ${atc3_tmp} (${atc3_link}/${length(filtered_atc3_list)}):'))
      
      tbl_tmp <- check_and_harvest_table(filtered_atc3_list[atc3_link])
      
      if (nrow(tbl_tmp) > 0) {
        
        writeLines(str_interp('  └ Found non-zero table at ${atc3_tmp}, scraping...'))
        atc5_ddds <- bind_rows(atc5_ddds, tbl_tmp)
        rm(tbl_tmp)
        
        next
      }
      
      filtered_atc4_list <- get_child_atcs(atc3_tmp, main_link)

      for (atc4_link in seq_along(filtered_atc4_list)) {
        
        atc4_tmp <- str_extract(filtered_atc4_list[atc4_link], atc_regex$atc4)
        
        writeLines(str_interp('  └ Scraping ATC5s under ${atc4_tmp} (${atc4_link}/${length(filtered_atc4_list)})'))
        
        tbl_tmp <- check_and_harvest_table(filtered_atc4_list[atc4_link])
        
        if (nrow(tbl_tmp) > 0) {

          atc5_ddds <- bind_rows(atc5_ddds, tbl_tmp)
          rm(tbl_tmp)
          
          next
        }
        

        
      }
      
    }
    
  }
  
  return(atc5_ddds)
  
}

## Apply ----

n03_n05_ddds <- scrape_ddds_atc2(c('N03', 'N05'))

fwrite(n03_n05_ddds, 'prepped_data/n03_n05_DDDs.csv')
