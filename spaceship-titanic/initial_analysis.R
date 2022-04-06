## Load packages ----
invisible({

  # Packages to load are found in the character vector below
  packages <- c(
    "tidyverse", "data.table", "ggplot2", 
    "scales", "tictoc", "infotheo", 
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
  setwd('E:/Projects/kaggle-analyses/spaceship-titanic') # PC
} else {
  setwd('~/Documents/Projects/kaggle/spaceship-titanic') # Macbook
}

## Introduction ----
# This is a solution to the Kaggle: Spaceship Titanic competition https://www.kaggle.com/competitions/spaceship-titanic/overview
# The issue is a binary classification problem, determining whether a given individual has been transported to another dimension

## Load data & manipulate ----
train <- fread('raw-data/train.csv') %>% as_tibble()
test  <- fread('raw-data/test.csv') %>% as_tibble()

all   <- bind_rows(
  train %>% mutate(dataset = 'train'),
  test %>% mutate(dataset = 'test')
) 

## 1a. EDA - Categorical Variables ----
# Getting a glimpse of the data
#  We have 15 variables and 12,970 observations
glimpse(all)

# View cardinality of variables
all %>% 
  select(-dataset) %>% 
  sapply(n_distinct)

# See distinct values of all present categorical variables
#  Potentially interesting for feature engineering:
#   Passenger ID is in form gggg_pp, with gggg representing the travel group and pp representing passenger within group
#   Cabin is in the form deck/num/side. Side can be P (Port) or S (Starboard).
#   Name can be split into first and last name, although using last name to derive family may not be much better than using group from Passenger ID
all %>% 
  select(-dataset) %>% 
  select_if(is.character) %>% 
  sapply(unique)

# Charts to see distributions of low cardinality categorical variables and booleans
#  Wary of VIP as only a small percentage are VIPs - even smaller than percentage of NAs
all %>% 
  # Variables of interest
  select(HomePlanet, Destination, CryoSleep, VIP) %>% 
  gather(variable, value) %>% 
  # Calculating percentage distributions for each value
  group_by(variable, value) %>% 
  summarise(
    n = n(),
    .groups = 'drop'
  ) %>% 
  group_by(variable) %>% 
  mutate(percentage = n / sum(n)) %>% 
  ungroup() %>% 
  # Charting
  ggplot(aes(x = value, y = percentage, label = percent(percentage, accuracy = 0.1))) +
  geom_col() +
  facet_wrap(vars(variable), scales = 'free') +
  #  Label formatting
  labs(x = 'Value', y = 'Percent') +
  geom_text(
    position = position_dodge(width = .9),
    vjust = -0.5,
    size = 3
  ) + 
  scale_y_continuous(labels = percent)

# See how these 4 variables vary with the target value - these tables have row-wise percentages
#  People from Europa seem to be transported at a higher frequency than people from other planets
#  People from Earth seem to be transported at a lower frequency
table(train$HomePlanet, train$Transported) %>% prop.table(1)

#  People headed to 55 Cancri e seem to be transported at a higher frequency than the other destinations
table(train$Destination, train$Transported) %>% prop.table(1)

#  People in CryoSleep seem to be transported at a higher frequency than not
table(train$CryoSleep, train$Transported) %>% prop.table(1)

#  People in VIP tend to be transported less - however be wary of this variable as there is only a small pct of VIPs
table(train$VIP, train$Transported) %>% prop.table(1)

# Deeper dive into Cabins
#  Split into Deck/Num/Side - we'll discuss Num in the numeric variables section
all_cabin_spl <- all %>% 
  mutate(Cabin = na_if(Cabin, "")) %>% 
  separate(Cabin, c("Deck", "Num", "Side"), "/")

#  View cardinality
all_cabin_spl %>% 
  select(Deck, Num, Side) %>% 
  sapply(n_distinct)

#  Distinct values for categorical variables and distribution chart
all_cabin_spl %>% 
  select(Deck, Side) %>% 
  sapply(unique)

#   Passengers tend to be heavily weighted towards Decks F and G
all_cabin_spl %>% 
  # Variables of interest
  select(Deck, Side) %>% 
  gather(variable, value) %>% 
  # Calculating percentage distributions for each value
  group_by(variable, value) %>% 
  summarise(
    n = n(),
    .groups = 'drop'
  ) %>% 
  group_by(variable) %>% 
  mutate(percentage = n / sum(n)) %>% 
  ungroup() %>% 
  # Charting
  ggplot(aes(x = value, y = percentage, label = percent(percentage, accuracy = 0.1))) +
  geom_col() +
  facet_wrap(vars(variable), scales = 'free') +
  #  Label formatting
  labs(x = 'Value', y = 'Percent') +
  geom_text(
    position = position_dodge(width = .9),
    vjust = -0.5,
    size = 3
  ) + 
  scale_y_continuous(labels = percent)

#  See how deck and side correspond to being transported
train_cabin_spl <- all_cabin_spl %>% filter(dataset == 'train') %>% select(-dataset)

#   Decks B, C, E, T all seem to be transported at a varying rate - note though that T is a very small deck (0.1% of all passengers)
table(train_cabin_spl$Deck, train_cabin_spl$Transported) %>% prop.table(1)

#   Doesn't look like Side matters too much here
table(train_cabin_spl$Side, train_cabin_spl$Transported) %>% prop.table(1)

# Deeper dive into passenger ID and travel groups
#  Let's create a variable to see the size of an individual's travelling group
all_grp_sep <- all %>% 
  separate(PassengerId, c('GroupId', 'PassengerId'), "_")

all_grps <- all_grp_sep %>% 
  distinct(GroupId, PassengerId) %>% 
  group_by(GroupId) %>% 
  summarise(
    group_size = n(), 
    .groups = 'drop'
  )

#  View distribution of group sizes
all_grps %>% 
  ggplot(aes(x = group_size)) +
  geom_histogram() +
  scale_x_continuous(breaks = unique(all_grps$group_size) %>% sort()) +
  scale_y_continuous(labels = comma) +
  labs(x = 'Group Size', y = "Count", title = "Distribution of Travel Group Sizes")

all_grouped <- all_grp_sep %>% 
  left_join(all_grps, by = 'GroupId')

train_grouped <- all_grouped %>% filter(dataset == 'train') %>% select(-dataset)

#  Looks like transportation rate varies a little based on travelling group size
table(train_grouped$group_size, train_grouped$Transported) %>% prop.table(1)

#  Determining whether the passenger is female or male using first name might be something for further review

## 1b. EDA - Numeric Variables ----
# The following numeric variables are available to us
all %>% 
  select(-dataset) %>% 
  select_if(is.numeric) %>% 
  summary()

#  As well as the room number of an individual
all_cabin_spl %>% 
  select(Num) %>% 
  mutate(Num = as.numeric(Num)) %>% 
  summary()

# View histograms to see if distribution changes by transportation status, indicating possibly useful features
#  Nothing seems to jump out as particularly illuminating, although it seems like younger children are more likely to be transported
train_cabin_spl %>% 
  mutate(Num = as.numeric(Num)) %>% 
  select_if(sapply(., is.numeric) | str_detect(names(.), "Transported")) %>% 
  gather(variable, value, -Transported) %>% 
  ggplot(aes(x = value, fill = Transported)) +
  geom_histogram(position = 'dodge') +
  facet_wrap(vars(variable), scales = 'free') +
  scale_y_continuous(labels = comma) +
  theme(legend.position = "bottom") + 
  labs(x = 'Value', y = 'Count')

## 2. Feature Importance ----
# While we've developed hypotheses regarding which features may have value in predicting whether an individual has been transported
#  or not, let's back these with some formal analysis.

all_wrk <- all_cabin_spl %>% 
  separate(PassengerId, c('GroupId', 'PassengerId'), "_") %>% 
  left_join(all_grps, by = 'GroupId') %>% 
  mutate_all(.funs = ~ na_if(., ""))

train_wrk <- all_wrk %>% filter(dataset == 'train') %>% select(-dataset)

# First we'll use mutual information to rank the variables in terms of information level
# train_wrk %>% 
#   select(-Transported) %>% 
#   lapply(function(x) mutinformation(., train_wrk$Transported))

mutual_info <- tibble(
  variable = train_wrk %>% select(-Transported, -GroupId, -Name, -PassengerId, -Num) %>% colnames(),
  mutual_info = NA_real_
)

for (i in 1:nrow(mutual_info)) {

  mutual_info[[i, 2]] <- mutinformation(train_wrk[mutual_info[[i, 1]]], train_wrk$Transported)
  
}

mutual_info_fct_order <- mutual_info %>% arrange(mutual_info) %>% pull(variable)

mutual_info <- mutual_info %>% 
  mutate(variable = factor(variable, levels = mutual_info_fct_order))

mutual_info %>% 
  ggplot(aes(x = mutual_info, y = variable)) +
  geom_col() +
  labs(x = 'Mutual Information', y = 'Variable')

# Note how the following potential candidates for features have low Mutual Information: HomePlanet, Destinatino, Deck, Age


# Next, we'll look at Information Value/Weight of Evidence which is related but slightly different: 
#  https://stats.stackexchange.com/questions/16945/why-do-people-use-the-term-weight-of-evidence-and-how-does-it-differ-from-poi
train_woe <- train_wrk %>% 
  select(-GroupId, -Name, -PassengerId) %>% 
  # Convert all character columns to factors
  mutate_if(
    is.character,
    factor
  ) %>% 
  # Force Transported to numeric
  mutate(Transported = as.numeric(Transported))

# Information values > 0.1 have medium strength
inf_value <- create_infotables(data = train_woe, y = "Transported", bins = 10)$Summary

# Comparing Mutual Information and Information Value
#  Generally the two measures agree in terms of ranking feature strength.
#  Information Values > 0.1 indicate a medium strength predictive power, so we'll start with these variables
grid.arrange(
  mutual_info %>% 
    ggplot(aes(x = mutual_info, y = variable)) +
    geom_col() +
    labs(x = 'Mutual Information', y = 'Variable'),
  inf_value %>% 
    mutate(Variable = factor(Variable, levels = inf_value %>% arrange(IV) %>% pull(Variable))) %>% 
    ggplot(aes(x = IV, y = Variable)) +
    geom_col() +
    labs(x = 'Information Value', y = element_blank()) +
    geom_vline(xintercept = 0.1),
  ncol = 2
)

## 3. Missing/NAs ----
all_wrk_tmp <- all_wrk

# Noting some missing values hidden as empty strings, convert to NAs and count number of missings
#  In total, 6,000 / 12,970 rows have at least one NA in them; definitely don't want to cut 50% of observations out of the dataset
all_wrk_tmp[rowSums(is.na(all_wrk_tmp %>% mutate_all(.funs = ~ na_if(., "")))) > 0, ] %>% nrow()

# Where do these missing values reside?
(all_wrk_tmp %>% 
  mutate_all(.funs = ~ na_if(., "")) %>% 
  sapply(function(x) sum(is.na(x))) / nrow(all_wrk_tmp)) %>% 
  as.list() %>% 
  data.frame() %>% 
  gather(variable, missing_pct) %>% 
  arrange(desc(missing_pct))

# Missing value handling
#  Name will not be used as a feature so we can leave this for now. Same as Age, VIP, and Destination.
#  The numeric values such as RoomService, FoodCourt, ShoppingMall, Spa, VRDeck will be predicted using mice
#  Cabin contains Deck which will need to be imputed - first determine if they're in a group and assign them to that room (assuming shared rooms)
#  HomePlanet - will see if this is related to Cabin at all
#  CryoSleep are confined to their quarters so will have likely spent no money on other amenities

# Will be using a mix of imputation and prediction here
# We'll focus on imputing values for features that we know we'll use in our model, or variables that
#  are used to impute those features.

# 3a. CryoSleep ----
#   Individuals in CryoSleep are confined to their Cabin throughout the duration of the interstellar trip,
#   this means that they are likely not to have spent any money on the additional luxury amenities onboard the ship.
all_cryo_sub <- all_wrk_tmp %>% 
  mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService)

all_cryo_sub %>% filter(CryoSleep & lux_spend > 0) %>% nrow() == 0 # No-one in CryoSleep has spent on amenities

# So we know that any rows with a non-zero sum of luxury spend will not be in CryoSleep

# Does this go the other way? No, not all people who spent nothing on luxuries were in CryoSleep, however a large
#  percentage were. After imputation, we can check if this is adhered to.
nonspender_cryosleep_rate <- all_wrk_tmp %>% 
  mutate(lux_spend_keepnas = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>% 
  filter(!is.na(CryoSleep) & lux_spend_keepnas == 0) %>% 
  group_by(CryoSleep) %>% 
  tally() %>% 
  mutate(pct = n / sum(n)) %>% 
  filter(CryoSleep) %>% 
  pull(pct)

all_wrk_tmp <- all_wrk_tmp %>% 
  mutate(
    lux_spend_ignorenas = rowSums(select(., ShoppingMall, VRDeck, FoodCourt, Spa, RoomService), na.rm = TRUE),
    CryoSleep = case_when(
      # This is the only subset we can infer with the information we have at the present
      #  People with 0 spend may have just chosen to spend nothing on extra amenities.
      is.na(CryoSleep) & lux_spend_ignorenas > 0 ~ FALSE,
      TRUE ~ CryoSleep
    )
  )

imput_cryosleep <- all_wrk_tmp %>%
  mutate(
    HomePlanet_fct = factor(HomePlanet), 
    Destination_fct = factor(Destination), 
    Deck_fct = factor(Deck)
  ) %>%
  select(CryoSleep, HomePlanet_fct, Destination_fct, Deck_fct, ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>%
  mice(method = 'rf', seed = 24601, maxit = 1) %>%
  complete() %>% 
  mutate(CryoSleep = as.logical(CryoSleep)) %>% 
  select(-ends_with('_fct'))

# Note that imputation using a random forest resulted in a very similar proportion of 0-spenders being in CryoSleep
imput_cryosleep %>% 
  mutate(lux_spend_keepnas = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>% 
  filter(!is.na(CryoSleep) & lux_spend_keepnas == 0) %>% 
  group_by(CryoSleep) %>% 
  tally() %>% 
  mutate(pct = n / sum(n)) %>% 
  filter(CryoSleep) %>% 
  pull(pct)

# For now, add in the imputed values back to the data-frame
all_wrk_tmp <- all_wrk_tmp %>% 
  select(-CryoSleep) %>% 
  bind_cols(imput_cryosleep %>% select(CryoSleep))

# See if imputed values on aggregate change the distribution
all_wrk %>%
  filter(!is.na(CryoSleep)) %>% 
  group_by(CryoSleep) %>% 
  tally() %>% 
  mutate(dataset = 'Unmodified') %>% 
  bind_rows(
    all_wrk_tmp %>% 
      filter(!is.na(CryoSleep)) %>% 
      group_by(CryoSleep) %>% 
      tally() %>% 
      mutate(dataset = 'Imputed')
  ) %>% 
  group_by(dataset) %>% 
  mutate(pct = n / sum(n)) %>% 
  ungroup() %>% 
  ggplot(aes(x = dataset, fill = CryoSleep, y = pct, label = percent(pct, 0.1))) +
  geom_col(position = 'dodge') +
  scale_y_continuous(labels = percent) +
  labs(x = 'Dataset', y = 'Percent') +
  geom_text(
    position = position_dodge(width = .9),
    vjust = -0.5,
    size = 3
  ) +
  theme(legend.position = 'bottom')

# 3bi. Luxury Amenities part 1 - CryoSleeping Individuals ----
# We can then infer that all luxury amenities will be 0 for people in CryoSleep - this will fix luxury amenities for 513 observations
all_cryo_sub %>% filter(CryoSleep & is.na(lux_spend)) %>% nrow()

# We now replace all NAs in the luxury amenities with 0 where CryoSleep is TRUE
all_wrk_tmp <- all_wrk_tmp %>%
  mutate_at(
    .vars = vars(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService),
    .funs = ~ ifelse(CryoSleep & is.na(.), 0, .)
  )

# 3bii. Luxury Amenities part 2 - Non-CryoSleeping Individuals ----

# Viewing relationship to other variables
# all_wrk_tmp %>%
#   filter(!CryoSleep) %>%
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>%
#   ggplot(aes(y = lux_spend, x = Deck, fill = Side)) +
#   geom_boxplot() +
#   scale_y_continuous(labels = dollar)

# all_wrk_tmp %>%
#   filter(!CryoSleep) %>%
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>%
#   ggplot(aes(y = lux_spend, x = Destination, fill = HomePlanet)) +
#   geom_boxplot() +
#   scale_y_continuous(labels = dollar)

# all_wrk_tmp %>%
#   filter(!CryoSleep) %>%
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>%
#   ggplot(aes(y = lux_spend, x = as.factor(group_size))) +
#   geom_boxplot() +
#   scale_y_continuous(labels = dollar)

# all_wrk_tmp %>%
#   filter(!CryoSleep) %>%
#   mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService) %>%
#   mutate(Num = as.numeric(Num)) %>%
#   ggplot(aes(x = Num, y = lux_spend, colour = HomePlanet)) +
#   geom_point() +
#   scale_y_continuous(labels = dollar) +
#   scale_x_continuous(labels = comma)
#  Side note, HomePlanet appears to be related to room number a bit - Europa are all low ~<500 numbers


#   Amenities spend otherwise appears to be related to Destination, HomePlanet, Deck, attempt
#    imputation via random forest
all_wrk_noncryo <- all_wrk_tmp %>% filter(!CryoSleep | is.na(CryoSleep))
  
imput_lux_spend <- all_wrk_noncryo %>% 
  mutate(
    HomePlanet_fct = factor(HomePlanet), 
    Destination_fct = factor(Destination), 
    Deck_fct = factor(Deck)
  ) %>% 
  select(HomePlanet_fct, Destination_fct, Deck_fct, ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
  # No imputation method for the first three variables, then random forests for the rest
  mice(method = 'rf', seed = 24601, maxit = 1) %>% 
  complete() %>% 
  select(-ends_with('_fct'))

#  Compare imputed values vs. non-imputed (label rows with NAs with 'imputed' flag)
all_wrk_noncryo %>% 
  mutate(imputed = ifelse(
    is.na(ShoppingMall + VRDeck + FoodCourt + Spa + RoomService),
    TRUE, FALSE
  )) %>%
  select(-ShoppingMall, -VRDeck, -FoodCourt, -Spa, -RoomService) %>% 
  bind_cols(
    imput_lux_spend %>% 
      select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService)
  ) %>% 
  select(imputed, ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
  gather(variable, value, -imputed) %>% 
  ggplot(aes(x = value, fill = imputed)) +
  geom_histogram() +
  facet_grid(rows = vars(variable), scale = 'free') +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(labels = dollar) +
  labs(x = 'Value', y = 'Count', fill = 'Imputed?')

#   Add imputed values back to dataset
all_wrk_tmp <- all_wrk_tmp %>% 
  # Filter out !CryoSleep and is.na(CryoSleep)
  filter(CryoSleep) %>% 
  bind_rows(
    all_wrk_noncryo %>%
      select(-ShoppingMall, -VRDeck, -FoodCourt, -Spa, -RoomService) %>% 
      bind_cols(
        imput_lux_spend %>% 
          select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService)
      )
  ) %>% 
  mutate(lux_spend = ShoppingMall + VRDeck + FoodCourt + Spa + RoomService)

# Check distributions before and after 
all_wrk %>% 
  select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
  mutate(dataset = 'Unmodified') %>% 
  bind_rows(
    all_wrk_tmp %>% 
      select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
      mutate(dataset = 'Imputed')
  ) %>% 
  gather(variable, value, -dataset) %>% 
  ggplot(aes(x = value, fill = dataset)) +
  geom_histogram(position = 'dodge') +
  facet_grid(rows = vars(variable), scales = 'free') +
  theme(legend.position = 'bottom') +
  scale_x_continuous(limits = c(NA, 10000), labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(x = 'Value', y = 'Count', fill = 'Dataset')

# 3b. Inferring HomePlanet using Deck & Room Number ----
# Determine whether HomePlanet is related to Deck
#  From this we can reasonably infer that if in Decks A, B, C, or T, your HomePlanet was Europa
#  Further, it appears that Deck G is only inhabited by people from Earth
all_wrk_tmp %>% 
  group_by(HomePlanet, Deck) %>% 
  tally() %>% 
  ggplot(aes(x = Deck, y = n, fill = HomePlanet)) +
  geom_col(position = 'dodge')

# If passenger's Deck is A-C or T, fix HomePlanet to Europa
# If passenger's Deck is G, fix HomePlanet to Earth
all_wrk_tmp <- all_wrk_tmp %>% 
  mutate(
    HomePlanet = case_when(
      is.na(HomePlanet) & Deck %in% c('A', 'B', 'C', 'T') ~ 'Europa',
      is.na(HomePlanet) & Deck == 'G' ~ 'Earth',
      TRUE ~ HomePlanet
    )
  )

# Room number also seems related to HomePlanet
all_wrk_tmp %>% 
  mutate(Num = as.numeric(Num)) %>% 
  select(HomePlanet, Num) %>% 
  ggplot(aes(x = Num, fill = HomePlanet)) +
  geom_histogram(position = 'dodge')
  
# Finally, those within the same Group will likely have the same HomePlanet
all_wrk_tmp %>% 
  distinct(GroupId, HomePlanet) %>% 
  filter(!is.na(HomePlanet)) %>% 
  group_by(GroupId) %>% 
  summarise(n = n(), .groups = 'drop') %>% 
  filter(n > 1) %>% nrow() == 0 # Confirmed

all_wrk_grp_homes <- all_wrk_tmp %>% 
  filter(!is.na(HomePlanet)) %>% 
  distinct(GroupId, HomePlanet) %>% 
  rename(HomePlanet_supplement = HomePlanet)

all_wrk_tmp <- all_wrk_tmp %>% 
  left_join(all_wrk_grp_homes, by = c("GroupId")) %>% 
  mutate(HomePlanet = coalesce(HomePlanet, HomePlanet_supplement), .keep = 'unused')
  
# For all else, impute HomePlanet using room number and deck using mice
imput_homeplanet <- all_wrk_tmp %>% 
  mutate(
    HomePlanet_fct = factor(HomePlanet),
    Deck_fct = factor(Deck),
    Num_num = as.numeric(Num)
  ) %>% 
  select(HomePlanet_fct, Deck_fct, Num_num) %>% 
  mice(method = 'rf', seed = 24601, maxit = 1) %>% 
  complete() %>% 
  mutate(HomePlanet = as.character(HomePlanet_fct)) %>% 
  select(-ends_with('_fct'), -ends_with('_num'))

# Add these back to the dataset
all_wrk_tmp <- all_wrk_tmp %>% 
  select(-HomePlanet) %>% 
  bind_cols(imput_homeplanet)
  
# Check distribution of imputed
all_wrk %>% 
  group_by(HomePlanet) %>% 
  tally() %>% 
  mutate(pct = n / sum(n)) %>% 
  mutate(dataset = 'Unmodified') %>% 
  bind_rows(
    all_wrk_tmp %>% 
      group_by(HomePlanet) %>% 
      tally() %>% 
      mutate(pct = n / sum(n)) %>% 
      mutate(dataset = 'Imputed')
  ) %>% 
  ggplot(aes(x = HomePlanet, y = pct, fill = dataset, label = percent(pct, 0.1))) +
  geom_col(position = 'dodge') +
  scale_y_continuous(labels = percent) +
  theme(legend.position = 'bottom') +  
  labs(x = 'Home Planet', y = 'Percent', fill = 'Dataset') +
  geom_text(
    position = position_dodge(width = .9),
    vjust = -0.5,
    size = 3
  )

# 3c. Inferring Deck using Group ----
# Check if there are any groups between test and training - there are none
all_wrk_tmp %>% 
  distinct(GroupId, dataset) %>% 
  group_by(GroupId) %>% 
  tally() %>% 
  ungroup() %>% 
  filter(n > 1) %>% 
  arrange_all()

# Hypothesis is that, although groups don't stay in the same room, they tend to stay 
#  in decks that are adjacent to each other

# First, we check if everyone in the same group generally stay in the same room together
all_wrk_grps <- all_wrk_tmp %>% 
  filter(group_size > 1) %>% 
  select(GroupId, PassengerId, Deck, Num, Side) %>% 
  group_by(GroupId) %>% 
  mutate(count = n()) %>% 
  ungroup() %>% 
  arrange(GroupId, PassengerId)

# Checking to see if whole groups stay within the exact same room
#  Doesn't seem to occur - only pattern is larger groups are more likely to be spread across rooms
all_wrk_grps %>% 
  # All distinct combinations of GroupId and Cabin
  distinct(GroupId, Deck, Num, Side) %>% 
  group_by(GroupId) %>% 
  tally() %>% 
  # Adding back group_size variable
  left_join(all_grps, by = 'GroupId') %>% 
  mutate(n_grouping = ifelse(n == 1, 'one room', 'more than one room')) %>% 
  group_by(group_size, n_grouping) %>% 
  tally() %>% 
  ungroup() %>% 
  group_by(group_size) %>% 
  mutate(total = sum(n)) %>% 
  mutate(pct = n / total) %>% 
  ggplot(aes(x = group_size, y = pct, fill = n_grouping)) +
  geom_col(position = 'dodge')
  
# Check for Deck closeness within groups
#  Assumption here is that decks are in alphabetical order on the ship and therefore are closer
#   together
deck_refactor <- tibble(
  Deck = all_wrk_grps %>% filter(!is.na(Deck)) %>% distinct(Deck) %>% pull(Deck) %>% sort(),
  DeckNum = c(1:(all_wrk_grps %>% filter(!is.na(Deck)) %>% distinct(Deck) %>% pull(Deck) %>% length()))
)

all_wrk_grps %>% 
  left_join(deck_refactor, by = 'Deck') %>% 
  group_by(GroupId) %>% 
  summarise(
    range = max(DeckNum) - min(DeckNum),
    .groups = 'drop'
  ) %>% 
  ggplot(aes(x = range)) +
  geom_histogram() +
  scale_y_continuous(labels = comma) +
  labs(x = 'Deck Distance with Groups', y = 'Count')

# Any other relationships? VIP/spenders?
#  Slight relationship with number, tend to be smaller room numbers
all_wrk_tmp %>% 
  mutate(Num = as.numeric(Num)) %>% 
  ggplot(aes(x = Num, fill = VIP)) +
  geom_histogram()

#  Not really any relationship with Decks
all_wrk_tmp %>% 
  group_by(Deck, VIP) %>% 
  tally() %>% 
  ungroup() %>% 
  ggplot(aes(y = n, x = Deck, fill = VIP)) +
  geom_col()

# Any relationship with lux_spend? Not really
all_wrk_tmp %>% 
  ggplot(aes(x = lux_spend)) +
  geom_histogram() +
  facet_grid(rows = vars(Deck))

# Could perform some probability distribution stuff with the above distribution
# Imputing with mice and using GroupId doesn't really work though, we'll create a 'missing' bucket for now

all_wrk_tmp <- all_wrk_tmp %>% 
  mutate(Deck = ifelse(is.na(Deck), 'NA', Deck))

# 3d. Final Rollup ----
# Check for NAs in our features of interest
all_wrk_tmp %>% 
  select_at(vars(inf_value %>% filter(IV > 0.1) %>% .$Variable)) %>% 
  sapply(function(x) sum(is.na(x))) # All good

# Impute some further variables using mean in case we'll need them later
imput_age <- all_wrk_tmp %>% 
  select(-GroupId, -PassengerId, -Name, -Transported, -dataset, -Num) %>% 
  # Convert all character columns to factors
  mutate_if(
    is.character,
    factor
  ) %>%
  mice(method = 'rf', seed = 24601, maxit = 1) %>% 
  complete()

all_wrk_tmp <- all_wrk_tmp %>% 
  select(-Age) %>% 
  bind_cols(imput_age %>% select(Age))
  
imput_homeplanet <- all_wrk_tmp %>% 
  mutate(
    HomePlanet_fct = factor(HomePlanet),
    Deck_fct = factor(Deck),
    Num_num = as.numeric(Num)
  ) %>% 
  select(HomePlanet_fct, Deck_fct, Num_num) %>% 
  mice(method = 'rf', seed = 24601, maxit = 1) %>% 
  complete() %>% 
  mutate(HomePlanet = as.character(HomePlanet_fct)) %>% 
  select(-ends_with('_fct'), -ends_with('_num'))
# Check that we haven't lost anyone
identical(
  all_wrk %>% distinct(GroupId, PassengerId) %>% arrange_all(),
  all_wrk_tmp %>% distinct(GroupId, PassengerId) %>% arrange_all()
) # True, all good.

# Finally, convert features of interest and response into factors unless they're numeric
all_wrk_final <- all_wrk_tmp %>% 
  mutate_at(
    .vars = vars(Transported, CryoSleep, Deck, HomePlanet),
    .funs = factor
  )
  
## 4. Training ----
# 4a. Splitting into train/validate/test ----
# Split back into training and test sets
test_clean <- all_wrk_final %>% filter(dataset == 'test') %>% select(-dataset)
train_clean <- all_wrk_final %>% filter(dataset == 'train') %>% select(-dataset)

# Split train_clean dataset into training and validation sets - ensure people in the 
#  same group are either in one or the other dataset
set.seed(24601)

train_val_split <- train_clean %>% 
  group_by(GroupId) %>% 
  # Get count of rows under each GroupId
  tally() %>% 
  # Generate a random number for each row which will be compared against the 
  #  probability of being in the test set (70%).
  mutate(rand = runif(nrow(.))) %>% 
  mutate(grouping = ifelse(rand < 0.7, 'train', 'validation'), .keep = 'unused')

# Confirm this preserves the ~70/30 split we've aimed for:
train_val_split %>% 
  group_by(grouping) %>% 
  summarise(n = sum(n), .groups = 'drop') %>% 
  mutate(pct = percent(n / sum(n), 0.1))

train_new <- train_clean %>% filter(GroupId %in% (train_val_split %>% filter(grouping == 'train') %>% .$GroupId))
valid_new <- train_clean %>% filter(GroupId %in% (train_val_split %>% filter(grouping == 'validation') %>% .$GroupId))

# 4b. Random Forest ----
# 4bi. Fitting process ----
set.seed(24601)

st_rf_mod <- randomForest(
  as.formula(
    paste('Transported', "~", c(inf_value %>% filter(IV > 0.1) %>% .$Variable, 'Age') %>% paste(collapse = " + "))
  ),
  data = train_new
)

# Plot OOB error
plot(st_rf_mod)
legend('topright', colnames(st_rf_mod$err.rate), col=1:3, fill=1:3)

# Plot variable importance
importance_output <- importance(st_rf_mod)
importance <- tibble(
  var = row.names(importance_output), 
  imp = round(importance_output[ ,'MeanDecreaseGini'],2)
)

importance %>% 
  mutate(var = factor(var, levels = importance %>% arrange(imp) %>% .$var)) %>% 
  ggplot(aes(x = imp, y = var)) +
  geom_col() +
  labs(x = 'Importance', y = 'Variable')
  
# 4bii. Validation ----
predicted_values <- valid_new %>% 
  bind_cols(
    Prediction = predict(st_rf_mod, valid_new)
  )

# Confusion matrix and accuracy metrics
prediction_cm <- confusionMatrix(
  predicted_values$Prediction, predicted_values$Transported)

prediction_cm$table
prediction_cm$byClass[['Sensitivity']]
prediction_cm$byClass[['Specificity']]

# Sub-groups
#  For now we'll look at the sub-group of VIPs, since there aren't many, we're not expecting the model
#  to have done particularly well here
prediction_vip_cm <- confusionMatrix(
  (predicted_values %>% filter(VIP))$Prediction, (predicted_values %>% filter(VIP))$Transported
)

prediction_vip_cm$table
prediction_vip_cm$byClass[['Sensitivity']]
prediction_vip_cm$byClass[['Specificity']]

# 4c. XGBoost ----
# Not XGBoost only works with numeric vectors, we'll modify our three datasets so that factors are one-hot encoded
#  and logicals are 0-1 (only for the variables of interest though)

# Define a function pipeline to clean all three data-frames
xgb_prep <- function(data, iv_threshold = 0.1) {
  
  # Bin Age and filter only for variables we're interested in
  data_tmp <- data %>% 
    mutate(AgeGrp = cut(Age, breaks = 10 * c(-1:10))) %>% 
    select_at(vars(inf_value %>% filter(IV > iv_threshold) %>% .$Variable, AgeGrp, Transported))
  
  # Force logical to numeric
  data_tmp <- data_tmp %>% 
    mutate_at(
      .vars = vars(CryoSleep#, Transported
                   ),
      .funs = ~ as.numeric(.) - 1
    ) %>% 
    # Label needs to be a factor to show this is a classification problem
    mutate(Transported = as.factor(Transported)) 
  
  # One-hot encode Deck and HomePlanet
  dummy_var_model <- dummyVars(~ Deck + HomePlanet + AgeGrp, data = data_tmp)
  
  # Add back to main dataset
  data_tmp <- data_tmp %>% 
    select(-Deck, -HomePlanet, -AgeGrp) %>% 
    bind_cols(
      predict(dummy_var_model, newdata = data_tmp)
    )
  
  return(data_tmp)
  
}

train_xgb <- xgb_prep(train_new)
valid_xgb <- xgb_prep(valid_new)
test_xgb <- xgb_prep(test_clean)

# 4ci. Fitting process ----
# First, define the controls we want to train with; we're choosing 10-fold cross-validation and a grid search
xgb_control <- trainControl(
  method = "cv", 
  number = 5, 
  search = "grid"
)

# Next, listing the possible hyperparameters we'll train over
#  For hyperparameters not listed here, we'll use the default value
xgb_hyp_params <- expand.grid(
  max_depth = c(3, 4, 5, 6), # Controls the max depth of each tree; higher values = more chance of overfitting 
  nrounds = c(1:15) * 50, # Number of trees to go through
  eta = c(0.01, 0.1, 0.2), # Analogous to learning rate
  gamma = c(0, 0.01, 0.1), # The minimum loss reduction required to split the next node
  
  # Default values for remaining hyperparameters
  subsample = c(0.5, 0.75, 1),
  min_child_weight = 1,
  colsample_bytree = 0.6
)

# Unregister any parallel workers
env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)

set.seed(24601)

# Training the model
st_xgb_mod <- train(
  Transported ~ ., 
  data = train_xgb, 
  method = "xgbTree", 
  trControl = xgb_control, 
  tuneGrid = xgb_hyp_params
)

xgb.plot.importance(
  xgb.importance(
    colnames(train_xgb %>% select(-Transported)), 
    model = st_xgb_mod$finalModel
  )
)

# 4cii. Validation
predicted_values_xgb <- valid_new %>% 
  bind_cols(
    Prediction = predict(st_xgb_mod, valid_xgb)
  )

# Confusion matrix and accuracy metrics
prediction_cm_xgb <- confusionMatrix(
  predicted_values_xgb$Prediction, predicted_values_xgb$Transported
)

prediction_cm_xgb$table
prediction_cm_xgb$byClass[['Sensitivity']]
prediction_cm_xgb$byClass[['Specificity']]

# Sub-groups
#  For now we'll look at the sub-group of VIPs, since there aren't many, we're not expecting the model
#  to have done particularly well here
prediction_vip_cm_xgb <- confusionMatrix(
  (predicted_values_xgb %>% filter(VIP))$Prediction, (predicted_values_xgb %>% filter(VIP))$Transported
)

prediction_vip_cm_xgb$table
prediction_vip_cm_xgb$byClass[['Sensitivity']]
prediction_vip_cm_xgb$byClass[['Specificity']]

# 5. Prediction ----
final_output <- test_clean %>% 
  unite('PassengerId', GroupId:PassengerId, sep = "_") %>% 
  select(PassengerId) %>% 
  bind_cols(
    Transported_fct = predict(st_rf_mod, test_clean)
  ) %>% 
  mutate(Transported = ifelse(Transported_fct == 'TRUE', 'True', 'False'), .keep = 'unused')

nrow(final_output) == 4277 # Size Kaggle expects for this solution

fwrite(final_output, 'output/spaceship_titanic_rf_solution.csv') # Score: 0.79565

final_output_xgb <- test_clean %>% 
  unite('PassengerId', GroupId:PassengerId, sep = "_") %>% 
  select(PassengerId) %>% 
  bind_cols(
    Transported_fct = predict(st_xgb_mod, test_xgb)
  ) %>% 
  mutate(Transported = ifelse(Transported_fct == 'TRUE', 'True', 'False'), .keep = 'unused')

nrow(final_output_xgb) == 4277 # Size Kaggle expects for this solution

fwrite(final_output_xgb, 'output/spaceship_titanic_rf_solution_xgb.csv') # Score: 0.79191

## X. Archive ----

#   Compare imputed values vs. non-imputed
# bind_rows(
#   all_wrk_noncryo %>% 
#     select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
#     gather(var, value) %>% 
#     mutate(dataset = 'Non-Imputed'),
#   imput_lux_spend %>% 
#     select(ShoppingMall, VRDeck, FoodCourt, Spa, RoomService) %>% 
#     gather(var, value) %>% 
#     mutate(dataset = 'Imputed')
# ) %>% 
#   ggplot(aes(x = value)) +
#   geom_histogram() +
#   facet_grid(rows = vars(var), cols = vars(dataset), scales = 'free') +
#   scale_x_continuous(labels = dollar) +
#   scale_y_continuous(labels = comma) +
#   labs(x = 'Spend', y = 'Count')

# 3b. Inferring room number using group

# # Checking to see if groups stay within the same side
# all_wrk_grps %>% 
#   # All distinct combinations of GroupId and Side
#   distinct(GroupId, Side) %>% 
#   group_by(GroupId) %>% 
#   tally() %>% 
#   # Adding back group_size variable
#   left_join(all_grps, by = 'GroupId') %>% 
#   mutate(n_grouping = ifelse(n == 1, 'one side', 'more than one side')) %>% 
#   group_by(group_size, n_grouping) %>% 
#   tally() %>% 
#   ungroup() %>% 
#   group_by(group_size) %>% 
#   mutate(total = sum(n)) %>% 
#   mutate(pct = n / total)

# all_wrk_tmp %>% 
#   filter(!is.na(CryoSleep)) %>% 
#   group_by(HomePlanet, Destination, CryoSleep) %>% 
#   tally() %>% 
#   group_by(HomePlanet, Destination) %>% 
#   mutate(
#     total = sum(n),
#     pct = n / total
#   ) %>% 
#   ungroup() %>% 
#   filter(CryoSleep) %>% 
#   ggplot(aes(x = HomePlanet, y = pct, fill = Destination)) +
#   geom_col(position = 'dodge')

# # Sort out the CryoSleep values that couldn't be imputed in previously
# CryoSleep = case_when(
#   is.na(CryoSleep) & lux_spend > 0 ~ FALSE,
#   # Use the rate of cryosleeping for 0-spenders to randomly pick who will be in cryosleep
#   is.na(CryoSleep) & !is.na(lux_spend) ~ ifelse(runif(1) <= nonspender_cryosleep_rate, TRUE, FALSE),
#   TRUE ~ CryoSleep
# )

# # Fr all else - Infer using mice random forest and check if it has adhered to this range
# tmp <- all_wrk_tmp %>%
#   mutate(
#     HomePlanet_fct = factor(HomePlanet),
#     Deck_fct = factor(Deck),
#     GroupId_num = as.numeric(GroupId)
#   ) %>%
#   select(Deck_fct, HomePlanet_fct, GroupId_num) %>%
#   mice(method = c('pmm', '', ''), seed = 24601, maxit = 1) %>%
#   complete()
# 
# tmp %>%
#   mutate(Deck = as.character(Deck_fct)) %>%
#   left_join(deck_refactor, by = 'Deck') %>%
#   group_by(GroupId_num) %>%
#   summarise(
#     range = max(DeckNum) - min(DeckNum),
#     .groups = 'drop'
#   ) %>%
#   ggplot(aes(x = range)) +
#   geom_histogram() +
#   scale_y_continuous(labels = comma) +
#   labs(x = 'Deck Distance with Groups', y = 'Count')

# # There are some values in the features of interest that couldn't be imputed by mice earlier, let's go over them again now
# all_wrk_tmp <- all_wrk_tmp %>% 
#   select(-starts_with('lux_spend'))
# 
# imput_all <- all_wrk_tmp %>% 
#   select(
#     Destination, Age, VIP, Transported, group_size, 
#     RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, 
#     CryoSleep, HomePlanet
#   ) %>% 
#   # Convert all character columns to factors
#   mutate_if(
#     is.character,
#     factor
#   ) %>% 
#   # Current a bug in versions of mice < 3.14.2 that breaks using
#   #  rf on columns with only one NA
#   # https://stackoverflow.com/a/70246986/17569265
#   mice(method = 'cart') %>% 
#   complete()
